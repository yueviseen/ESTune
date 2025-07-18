# -*- coding: gbk -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import random
import numpy as np

# ---- 1. Define a Feedforward Neural Network (FNN) for encoding static features ----
class FNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FNN, self).__init__()
        # First fully connected (linear) layer: from input to first hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # First activation function (ReLU) after the first hidden layer
        self.relu1 = nn.ReLU()
        # Second fully connected layer: from first hidden to second hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Second activation function (ReLU) after the second hidden layer
        self.relu2 = nn.ReLU()
        # Third fully connected layer: from second hidden to output layer
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Pass input through the first linear layer
        x = self.fc1(x)
        # Apply first ReLU activation
        x = self.relu1(x)
        # Pass through the second linear layer
        x = self.fc2(x)
        # Apply second ReLU activation
        x = self.relu2(x)
        # Pass through the third linear layer (output layer)
        x = self.fc3(x)
        # Return the final output
        return x


# ---- 2. Bayesian Linear Layer Definition ----
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        """
        A Bayesian Linear layer where the weights and biases are modeled as distributions
        instead of deterministic values.
        """
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight and bias mean and log-variance parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))

        self.weight = None
        self.bias = None

    def sample_weight(self):
        """
        Sample weights and biases using the reparameterization trick for variational inference.
        """
        sigma_weight = torch.log1p(torch.exp(self.weight_rho))
        epsilon_weight = Normal(0, 1).sample(self.weight_mu.shape)
        weight = self.weight_mu + sigma_weight * epsilon_weight

        sigma_bias = torch.log1p(torch.exp(self.bias_rho))
        epsilon_bias = Normal(0, 1).sample(self.bias_mu.shape)
        bias = self.bias_mu + sigma_bias * epsilon_bias

        return weight, bias

    def forward(self, x):
        weight, bias = self.sample_weight()
        return nn.functional.linear(x, weight, bias)

# ---- 3. Four-layer Bayesian Neural Network ----
class BayesianFNN4Layer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Four-layer Bayesian Feedforward Neural Network.
        Structure: Input -> Hidden1 -> Hidden2 -> Hidden3 -> Output.
        All layers are Bayesian (uncertainty-aware).
        """
        super(BayesianFNN4Layer, self).__init__()
        # First Bayesian fully-connected (linear) layer
        self.bayesian_fc1 = BayesianLinear(input_dim, hidden_dim)
        # Second Bayesian fully-connected layer
        self.bayesian_fc2 = BayesianLinear(hidden_dim, hidden_dim)
        # Third Bayesian fully-connected layer
        self.bayesian_fc3 = BayesianLinear(hidden_dim, hidden_dim)
        # Output Bayesian fully-connected layer
        self.bayesian_fc4 = BayesianLinear(hidden_dim, output_dim)
        # Activation function used between layers (ReLU is common)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through each layer with ReLU activations between hidden layers.
        No activation on the output layer for regression tasks.
        """
        x = self.relu(self.bayesian_fc1(x))   # Pass through first Bayesian layer and apply ReLU
        x = self.relu(self.bayesian_fc2(x))   # Pass through second Bayesian layer and apply ReLU
        x = self.relu(self.bayesian_fc3(x))   # Pass through third Bayesian layer and apply ReLU
        x = self.bayesian_fc4(x)              # Final Bayesian layer (output layer), no activation
        return x

# ---- 4. Hybrid Bayesian Neural Network (FNN + GRU+ BNN) ----
class SimpleGRUWithBayesianOutput(nn.Module):
    def __init__(self, input_size, mid_size1, mid_size2, output_size, num_layers, static_input_size, fnn_hidden_size, bnn_hidden_size):
        """
        Hybrid model:
        - GRU processes sequence (e.g., time-series segment data).
        - FNN encodes static features (e.g., configuration vectors).
        - Output is fused and passed through Bayesian Linear layer.
        """
        super(SimpleGRUWithBayesianOutput, self).__init__()
        self.gru = nn.GRU(input_size, mid_size1, num_layers=num_layers, batch_first=True)
        self.fnn_static = FNN(static_input_size, fnn_hidden_size, mid_size2)
        self.fc_final = BayesianFNN4Layer(mid_size1 + mid_size2, bnn_hidden_size, output_size)

    def forward(self, x, static_input):
        """
        x: [seq_len, batch, input_size] — sequence data for GRU
        static_input: [batch, static_input_size] — static features
        """
        output, _ = self.gru(x)             # output: [seq_len, batch, mid_size1]
        output = output[-1, :, :]           # Take last time step (last in sequence) [batch, mid_size1]
        static_input = self.fnn_static(static_input)  # Encode static features [batch, mid_size2]
        combined = torch.cat((output, static_input), dim=1)  # [batch, mid_size1 + mid_size2]
        output = self.fc_final(combined)    # Bayesian output
        return output

# ---- 5. Custom Loss Function: ELBO (Evidence Lower Bound) ----
def elbo_loss(y_pred, y_true, model):
    """
    Combines negative log-likelihood, KL divergence, and mean squared error for Bayesian NN.
    """
    likelihood = Normal(y_pred, 1)
    nll = -likelihood.log_prob(y_true).sum()  # Negative log-likelihood

    kl_divergence = 0
    for module in model.modules():
        if isinstance(module, BayesianLinear):
            kl_divergence += torch.sum(torch.log(module.weight_rho.exp()) - module.weight_rho +
                                       0.5 * module.weight_mu.pow(2) +
                                       0.5 * module.weight_rho.exp().pow(2) - 0.5)
            kl_divergence += torch.sum(torch.log(module.bias_rho.exp()) - module.bias_rho +
                                       0.5 * module.bias_mu.pow(2) +
                                       0.5 * module.bias_rho.exp().pow(2) - 0.5)
    mse = torch.mean((y_pred - y_true) ** 2)
    return nll + kl_divergence + mse

# # ---- 7. Data Normalization Helpers ----
# def max_min(x, max, min):
#     """Normalize a scalar to [0, 1] given min/max."""
#     return (x-min)/(max-min)
#
# def max_min_list(x, max, min):
#     """Normalize a list of scalars to [0, 1]."""
#     return [(i-min)/(max-min) for i in x]
#
# # ---- 8. Synthetic Data Generator ----
# def generater_data(num):
#     """
#     Generate synthetic data.
#     - Each sample: input sequence + static feature + target
#     - Example: target = x1 * 100 + s1 - s2
#     """
#     input_list, output_list, static_list = [], [], []
#     for i in range(num):
#         min_val, max_val = 1, 100
#         seq_len = random.randint(1,4)
#         mid_list = [random.randint(min_val, max_val) for _ in range(seq_len)]
#         si_1 = random.randint(1, 100)
#         si_2 = random.randint(1, 100)
#         out =  mid_list[0] * 100 + si_1 - si_2
#
#         mid_list = max_min_list(mid_list, 100, 1)
#         si_1 = max_min(si_1, 100, 1)
#         si_2 = max_min(si_2, 100, 1)
#         out = max_min(out, 10099, 1)
#         input_list.append(mid_list)
#         static_list.append([si_1, si_2])
#         output_list.append(out)
#
#     inputs = [torch.tensor([[x] for x in sublist]).float() for sublist in input_list]  # [seq_len, 1]
#     targets = [torch.tensor([value]).float() for value in output_list]                # [1]
#     static_inputs = [torch.tensor(values).float() for values in static_list]          # [2]
#     return (inputs, targets, static_inputs)

# # ---- 9. Model Parameter Settings ----
# input_size = 1            # Each step is a scalar
# fnn_hidden_size = 32      # Hidden size for FNN
# mid_size1 = 1             # GRU output size
# mid_size2 = 8             # FNN output size
# output_size = 1           # Regression
# num_layers = 1
# static_input_size = 100     # Two static features
# bnn_hidden_size = 8

# ---- 6. Build Model ----
# model = SimpleGRUWithBayesianOutput(input_size, mid_size1, mid_size2, output_size, num_layers, static_input_size, fnn_hidden_size, bnn_hidden_size)
# criterion = elbo_loss


 # Number of epochs for training

# ---- 6. Standard Training Function ----
def train(model, data, optimizer, criterion, num_epochs=50):
    """
    Standard training loop for the hybrid model.
    """
    num_samples = 10  # Number of posterior samples per prediction
    (inputs, targets, static_inputs) = data
    for epoch in range(num_epochs):
        total_loss = 0.0
        for x, y, static_input in zip(inputs, targets, static_inputs):
            optimizer.zero_grad()
            x = x.unsqueeze(1)           # [seq_len, 1, 1] for GRU
            static_input = static_input.unsqueeze(0)  # [1, static_input_size]
            y_preds = []
            for _ in range(num_samples):
                y_pred = model(x, static_input)
                y_preds.append(y_pred)
            y_pred = torch.stack(y_preds).mean(dim=0)
            loss = criterion(y_pred, y.unsqueeze(1), model)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(inputs):.4f}')

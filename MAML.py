# -*- coding: gbk -*-
import torch
import torch.nn as nn
import torch.optim as optim
from pycparser.ply.yacc import resultlimit
from torch.distributions import Normal
import random
import numpy as np
from HBNN import *
import os
import pandas as pd
import json
from Get_histotical_data import *
from Setting import *
from Get_current_data import *
from Get_result import *

# MAML (Model-Agnostic Meta-Learning) Implementation
class MAML:
    def __init__(self, model):
        """
        Initialize the MAML instance with a neural network model.
        :param model: The base learner (neural network) to be meta-trained.
        """
        self.model = model
        self.meta_optimizer = optim.Adam(model.parameters(), lr=0.001)  # Meta-optimizer for meta-update steps

    def inner_update(self, model, data, lr_inner=0.01, num_updates=1):
        """
        Perform the inner-loop adaptation for a single task.
        :param model: The model clone to adapt on a specific task.
        :param data: Task data, typically (inputs, targets, static_inputs).
        :param lr_inner: Inner loop learning rate.
        :param num_updates: Number of inner update steps.
        """
        optimizer = optim.Adam(model.parameters(), lr=lr_inner)
        (inputs, targets, static_inputs) = data
        for _ in range(num_updates):
            total_loss = 0
            for x, y, static_input in zip(inputs, targets, static_inputs):
                optimizer.zero_grad()
                x = x.unsqueeze(1)          # Add batch dimension for input
                static_input = static_input.unsqueeze(0)  # Add batch dimension for static input

                y_preds = []
                for _ in range(num_samples):  # num_samples should be defined externally
                    y_pred = model(x, static_input)
                    y_preds.append(y_pred)

                y_pred = torch.stack(y_preds).mean(dim=0)  # Average predictions for Bayesian output
                loss = criterion(y_pred, y.unsqueeze(1), model)  # criterion should be defined externally

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        print(f'Epoch, Loss: {total_loss / len(inputs)}')

    def meta_update(self, tasks, iterations=10, num_updates=1):
        """
        Perform meta-update across multiple tasks for several iterations.
        :param tasks: List of tasks, each with its own data (inputs, targets, static_inputs).
        :param iterations: Number of meta-training iterations.
        :param num_updates: Number of inner updates per task.
        """
        for _ in range(iterations):
            for task in tasks:
                data = task
                cloned_model = self.clone_model()  # Clone the base model for task-specific adaptation
                self.inner_update(cloned_model, data, lr_inner=0.001, num_updates=num_updates)

                # Accumulate gradients from task-adapted (cloned) model back to the base model
                for p, cloned_p in zip(self.model.parameters(), cloned_model.parameters()):
                    if p.grad is None:
                        p.grad = cloned_p.grad.clone()
                    else:
                        p.grad += cloned_p.grad.clone()
                self.meta_optimizer.step()  # Update meta-learner parameters

            # (Optional) Print parameter values for debugging/inspection
            for name, param in self.model.named_parameters():
                print(f"{name}: shape={param.shape}")
                print(param.data)

    def clone_model(self):
        """
        Clone the model structure and parameters for task-specific adaptation.
        This implementation is specific to SimpleGRUWithBayesianOutput class.
        Modify the instantiation as per your model class and constructor arguments.
        """
        model_clone = SimpleGRUWithBayesianOutput(input_size, mid_size1, mid_size2, output_size, num_layers, static_input_size, fnn_hidden_size, bnn_hidden_size)
        model_clone.load_state_dict(self.model.state_dict())
        return model_clone



if __name__ == '__main__':
    # --------------------- Parameter Setup ---------------------
    # Define neural network architecture parameters
    input_size = 1  # Each step is a scalar
    fnn_hidden_size = 32  # Hidden size for FNN
    mid_size1 = 1  # GRU output size
    mid_size2 = 8  # FNN output size
    output_size = 1  # Regression
    num_layers = 1 # Single-layer GRU
    static_input_size = 100  # Two static features
    bnn_hidden_size = 8 # Hidden size for BNN


    # Instantiate the HBNN model (GRU + FNN + Bayesian output layer)
    model = SimpleGRUWithBayesianOutput(input_size, mid_size1, mid_size2, output_size, num_layers, static_input_size, fnn_hidden_size, bnn_hidden_size)

    # Define the loss function (ELBO: MSE + Bayesian KL divergence)
    criterion = elbo_loss
    global num_samples
    num_samples = 5    # Number of Monte Carlo samples for Bayesian output estimation

    # Collect relevant data from historical tasks.
    task_list = load_all_tasks_from_dir(historical_task_data)

    # ----------------- Meta-Training with MAML -----------------
    maml = MAML(model)
    print("Start meta-training...")
    # For each iteration, perform meta-update across all tasks
    maml.meta_update(task_list, iterations=10, num_updates=1)
    print("Meta-training finished.")

    # ----------------- HBNN initialization completed. -----------------
    # Clone the base meta-trained model for task-specific adaptation
    cloned_model = maml.clone_model()


    # 获取当前任务的数据
    new_task_list = load_task_from_json(curreent_task_data)

    # Conduct model updating
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    if len(new_task_list[0]) >= 20:
        result = read_performance_from_json(current_task_interation_result_data)
        prefixes = [new_task_list[0][:i] for i in range(1, len(new_task_list[0]) + 1)]
        for prefix in prefixes:
            train_data = task_list[prefix], result, new_task_list[1][1]
            train(model, train_data, optimizer, criterion, num_epochs=10)

    # Perform performance prediction
    else:
        for task_idx, new_task in enumerate(new_task_list):
            # Unpack the data for the current task
            inputs, static_inputs = new_task

            # Select the last sample of the current task for prediction
            x, s = inputs[-1], static_inputs[-1]

            with torch.no_grad():
                # Reshape the sequence input to match GRU input requirements: [seq_len, batch=1, feature_dim=1]
                x_ = x.unsqueeze(1)

                # Reshape the static input to add a batch dimension: [batch=1, static_feature_dim]
                s_ = s.unsqueeze(0)

                # Perform multiple stochastic forward passes through the Bayesian model
                # Each pass samples from the weight distributions, producing a slightly different output
                y_preds = [cloned_model(x_, s_) for _ in range(num_samples)]  # Monte Carlo sampling

                # Stack predictions to a single tensor of shape [num_samples, 1, 1] (or similar)
                y_preds = torch.stack(y_preds)

                # Compute the mean of the predictions (Bayesian predictive mean)
                mean_pred = y_preds.mean(dim=0).item()

                # Compute the standard deviation (Bayesian predictive uncertainty)
                std_pred = y_preds.std(dim=0).item()

                # Output the true value, predicted mean, and predictive standard deviation for this last sample
                print(
                    f"Task {task_idx + 1} - Last Sample: "
                    f"Predicted Mean={mean_pred:.4f}, "
                    f"Std={std_pred:.4f}"
                )



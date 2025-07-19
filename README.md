# ESTune: Bayesian Uncertainty-Guided Early Stopping for Database Configuration Tuning.

## The core code of ESTune is primarily composed of the six Python files mentioned above. The specific functionalities of each file are described as follows：

(1)Setting.py serves as a configuration module, primarily responsible for the centralized management and recording of storage paths for various types of data. Specifically, it includes:
historical_task_data_path, which specifies the storage location of historical tuning task data; current_task_data_path, which indicates the storage path for tuning task data generated during the current iteration; current_task_iteration_result_data_path, which records the storage location of database performance data obtained after the complete execution of the workload in the current iteration.

(2) Get_history_data.py is primarily used to store the tuning data of historical tasks, with the data from each tuning task saved individually in a file ending with the .json extension. Each JSON file contains multiple sample records, and each record consists of three components: seq, which represents the segment performance dataset; static, a vector composed of configuration parameter values; and target, which denotes the database performance metric corresponding to the given configuration. Example:
  ```
                  [
                    {"inputs": x1, "targets": y1, "static": s1},
                    {"inputs": x2, "targets": y2, "static": s2},
                    # ...
                  ]
```

(3) Get_current_data.py primarily used to store the data of the current iteration, with all data saved in a single file ending with the .json extension. This file contains multiple sample records, each consisting of two components: seq, which represents the segment performance dataset, and static, a vector composed of configuration parameter values. Example:
  ```
                  [
                    {"inputs": x1,  "static": s1},
                    {"inputs": x2,  "static": s2},
                    # ...
                  ]
```

(4) Get_result.py. If the workload is fully executed in the current iteration, the performance of the corresponding database configuration in this iteration is stored in this file. This file contains only a single component, performance. Example:
  ```
                   [
                    {"performanc": p1},
                   ]
```



(5) HBNN.py implements a hybrid neural network framework for performance prediction. It includes a feedforward neural network (FNN) for konb feature encoding, a Bayesian neural network with four Bayesian linear layers to capture uncertainty, and a hybrid model that integrates a GRU for sequence modeling with the FNN and Bayesian layers. The code supports custom ELBO loss computation for Bayesian learning, as well as standard training routines. The design enables robust prediction and uncertainty quantification for dynamic, variable-length input data, making it suitable for research and practical applications in performance analysis.

(6) MAML.py  implements a meta-learning-based Bayesian neural network framework for database knob tuning and performance prediction. The core meta-learning logic is realized through a MAML (Model-Agnostic Meta-Learning) class, which enables rapid adaptation to new database workloads using data from historical tuning tasks. The training pipeline includes both meta-training on historical task datasets and fast adaptation or prediction on current tasks. Bayesian inference is supported via Monte Carlo sampling to estimate both the predictive mean and uncertainty (standard deviation) for each configuration. 

The Operations_Database directory primarily contains code for fundamental database operations, such as checking the database running status, starting the database, and stopping the database. The Tuning_Model directory mainly includes the implementation of state-of-the-art tuning algorithms, such as SMAC and DDPG.

## Run Experiments 

After specifying the relevant paths in the Setting module, you can directly run the MAML.py file. Its specific workflow consists of the following steps:

（1）it first constructs a Hybrid Bayesian Neural Network. See line 110 in MAML.py: 
```
model = SimpleGRUWithBayesianOutput(input_size, mid_size1, mid_size2, output_size, num_layers, static_input_size, fnn_hidden_size, bnn_hidden_size)
```

(2)it then uses MAML to initialize the HBNN model. See line 124 in MAML.py:
```
maml.meta_update(task_list, iterations=10, num_updates=1)
``` 

(3) For the initialized HBNN, performance prediction is conducted using the latest data. See line 146~178 in MAML.py:
```
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

```

(4) If the workload is fully executed, the HBNN will be updated with the new data; otherwise, it will continue to perform performance prediction based on the existing model. See line 137~142 in MAML.py:
```
  if len(new_task_list[0]) >= 20:
      result = read_performance_from_json(current_task_interation_result_data)
      prefixes = [new_task_list[0][:i] for i in range(1, len(new_task_list[0]) + 1)]
      for prefix in prefixes:
          train_data = task_list[prefix], result, new_task_list[1][1]
          train(model, train_data, optimizer, criterion, num_epochs=10)
```

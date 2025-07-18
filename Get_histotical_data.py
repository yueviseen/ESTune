import os
import json
import torch

def load_task_from_json(filepath):
    """
    Load a single task from a JSON file.
    Returns:
        - inputs: list of tensors, each with shape [seq_len, 1]
        - targets: list of tensors, each with shape [1]
        - static_inputs: list of tensors, each with shape [static_dim]
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    inputs, targets, static_inputs = [], [], []
    for item in data:
        seq = torch.tensor(item['seq'], dtype=torch.float32).unsqueeze(1)  # [seq_len, 1]
        static = torch.tensor(item['static'], dtype=torch.float32)         # [static_dim]
        target = torch.tensor([item['target']], dtype=torch.float32)       # [1]
        inputs.append(seq)
        static_inputs.append(static)
        targets.append(target)
    return (inputs, targets, static_inputs)

def load_all_tasks_from_dir(data_dir):
    """
    Load all tasks from JSON files in the given directory.
    Each JSON file is considered a separate meta-learning task.
    Returns:
        - task_list: list of tasks, each in (inputs, targets, static_inputs) format
    """
    file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])
    task_list = []
    for fname in file_list:
        task_path = os.path.join(data_dir, fname)
        task_data = load_task_from_json(task_path)
        task_list.append(task_data)
    return task_list

# ------------------- Usage Example -------------------


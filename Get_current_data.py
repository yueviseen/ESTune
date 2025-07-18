import json
import torch

def load_inputs_from_json(filepath):
    """
    Load sequence and static features from a JSON file.
    Each entry should contain:
        - "seq": a list of floats (variable-length sequence)
        - "static": a list of floats (static features)
    Returns:
        - inputs: a list of tensors, each with shape [seq_len, 1]
        - static_inputs: a list of tensors, each with shape [static_dim]
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    inputs = []
    static_inputs = []
    for item in data:
        seq_tensor = torch.tensor(item['seq'], dtype=torch.float32).unsqueeze(1)  # [seq_len, 1]
        static_tensor = torch.tensor(item['static'], dtype=torch.float32)         # [static_dim]
        inputs.append(seq_tensor)
        static_inputs.append(static_tensor)
    return inputs, static_inputs


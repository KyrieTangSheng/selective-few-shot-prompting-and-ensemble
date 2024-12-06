import torch

def get_device(force_cpu=False):
    """
    Get the best available device.
    Args:
        force_cpu: If True, always return CPU (useful for FAISS operations)
    """
    if force_cpu:
        return torch.device("cpu")
        
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def to_numpy(tensor):
    """
    Safely convert tensor to numpy array, handling device transfers.
    """
    if tensor.device.type in ['cuda', 'mps']:
        tensor = tensor.cpu()
    return tensor.numpy()
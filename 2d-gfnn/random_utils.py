
from functools import wraps
import os, json
import torch
from collections.abc import Iterable

def resize_x_and_s(x: torch.Tensor, s: torch.Tensor):
    '''
    Resizes x (b x 2 Tensor) and s (f x 2 Tensor) to b x f x 2 Tensors.

    Parameters:
        x (torch.Tensor): b x 2 Tensor
        s (torch.Tensor): b x f x 2 | f x 2 Tensor
    
    Returns:
        x (b x f x 2 torch.Tensor), s (b x f x 2 torch.Tensor)
    '''
    assert x.dim() == 2, f"x ({x.shape}) should be of size b x 2."

    if s.dim() == 2:
        x_ret = x[:,None,:].expand(-1, s.shape[0], -1)
        s_ret = s[None, ...].expand(x.shape[0], -1, -1)
    
    elif s.dim() == 3:
        assert s.shape[0] == x.shape[0], f"if s ({s.shape}) is a 3D Tensor, it must have the same shape in dim 0 as x ({x.shape})."
        x_ret = x[:,None,:].expand(-1, s.shape[1], -1)
        s_ret = s

    assert x_ret.shape == s_ret.shape == (x.shape[0], s_ret.shape[1], 2)

    return x_ret, s_ret

    

def find_line_with_keyword(file_path, keyword, index: int = None):
    """
    Returns the first line in the file that starts with the given keyword.
    If no such line exists, returns None.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    if index is not None:
        line = open(file_path, "r").readlines()[index]
        if line.startswith(keyword):
            return line.rstrip('\n')
    else:
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith(keyword):
                    return line.rstrip('\n')
    raise ValueError(f"No line starting with '{keyword}' found in {file_path}.")


def log_dict_as_json(dict: dict, file_path: str):
    """
    Logs a dictionary as a JSON file.
    """
    with open(file_path, 'w') as f:
        json.dump(dict, f, indent=2)
    print(f"Logged dictionary to {file_path}")


def retrieve_dict_from_json(file_path: str) -> dict:
    """
    Retrieves a dictionary from a JSON file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    with open(file_path, 'r') as f:
        return json.load(f)
    


def apply_list(pos_idx: int):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Apply to the first argument only; adapt as needed
            pos_arg = args[pos_idx]
            if isinstance(pos_arg, Iterable) and not isinstance(pos_arg, (str, bytes)):
                return [func(a, *args[1:], **kwargs) for a in pos_arg]
            return func(*args, **kwargs)
        return wrapper
    return decorator
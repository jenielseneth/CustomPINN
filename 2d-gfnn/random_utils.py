
import os

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

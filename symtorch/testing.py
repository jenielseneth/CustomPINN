import sympy
import torch

from sympy.abc import x, y

expr = sympy.sin(x) + sympy.cos(y)

# Define a dictionary mapping SymPy functions to PyTorch
torch_namespace = {
    'sin': torch.sin,
    'cos': torch.cos,
    'exp': torch.exp,
    'log': torch.log,
    'sqrt': torch.sqrt,
    'Abs': torch.abs,
    'sign': torch.sign,
    'tan': torch.tan,
    'asin': torch.arcsin,
    'acos': torch.arccos,
    'atan': torch.arctan,
}

# Create a PyTorch-compatible function
f = sympy.lambdify([x, y], expr, modules=[torch_namespace])

# Evaluate on tensors
x_val = torch.tensor([1.0, 2.0])
y_val = torch.tensor([0.5, 1.5])

result = f(x_val, y_val)
print(result)
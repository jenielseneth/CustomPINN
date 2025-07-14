import math
import torch
import sympy
import random


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

x_sym, y_sym = sympy.symbols('x y')

def get_gaussian_expr(a, sigmas : tuple, means : tuple):
    assert len(sigmas) == len(means) == 2
    expr = a * sympy.exp(-((x_sym-means[0])**2/(2*sigmas[0]**2) + (y_sym-means[1])**2/(2*sigmas[1]**2)))
    return expr


def get_diffusion_term_a_expr():
    return 1
    return sympy.sin(x_sym*y_sym)

def generate_u_expr(n_expr: int = 5):
    u_exprs = []
    for i in range(n_expr):
        expr = (
            random.uniform(0.5, 5.0) * sympy.sin(random.randint(1, 5)*x_sym) * 
            sympy.cos(random.randint(1, 5)*y_sym) +
            random.uniform(0.1, 5.0) * x_sym**random.randint(1, 3) * y_sym**random.randint(1, 3)
        )
        # expr = (sympy.exp(-1/2*(20*(x_sym-0.5)**2 +20*(y_sym-0.5)**2))/(2*math.pi))
        u_exprs.append(expr)
    return u_exprs

def get_u_bnd_expr():
    expr = 0
    return expr

def generate_u_expr_w_bnd(domain, u_bnd_expr, n_expr: int = 5):
    '''
    Generate u(x) expressions based on boundary conditions. Currently only implemented for constant boundary conditions.
    '''
    x_min, x_max, y_min, y_max = domain
    u_exprs = []
    for i in range(n_expr):
        x_max_term = (x_sym-x_max)
        x_min_term = (x_sym-x_min)
        y_max_term = (y_sym-y_max)
        y_min_term = (y_sym-y_min)
        # if random.randint(1, 5) == 1:
        #     x_max_term = sympy.sin(x_max_term)
        #     x_min_term = sympy.sin(x_min_term)
        #     y_max_term = sympy.sin(y_max_term)
        #     y_min_term = sympy.sin(y_min_term)
        expr = (random.randint(1,100) * (x_max_term**random.randint(1, 5))*(x_min_term**random.randint(1, 5))
                * (y_max_term**random.randint(1, 5))*(y_min_term**random.randint(1, 5))
                + u_bnd_expr
        )
        u_exprs.append(expr)
    return u_exprs

def get_f_expr(u_exprs, a_expr):
    f_exprs = []
    for u in u_exprs:
        u_x = sympy.diff(u, x_sym)
        u_xx = sympy.diff(u, x_sym, 2)
        u_y = sympy.diff(u, y_sym)
        u_yy = sympy.diff(u, y_sym, 2)
        a_x = sympy.diff(a_expr, x_sym)
        a_y = sympy.diff(a_expr, y_sym)
        expr = -(u_x * a_x + a_expr * u_xx + u_y * a_y + a_expr * u_yy)
        f_exprs.append(expr)
    return f_exprs

def expr_to_func(exprs):
    if not type(exprs) == list:
        return sympy.lambdify([x_sym, y_sym], exprs, modules=[torch_namespace])
    
    funcs = [sympy.lambdify([x_sym, y_sym], expr, modules=[torch_namespace]) for expr in exprs]
    return funcs

def func_input_wrapper(funcs, device = None):
    '''
    changes functions that take x, y as input to taking points of form batch_size x 2 as input.

    Returns:
        changed_funcs (list): List of input functions.
    '''
    if not type(funcs) == list:
        return lambda points, f=funcs: torch.ones(len(points), device=device) * f(points[:, 0], points[:, 1])

    changed_funcs = [lambda points, f=f: torch.ones(len(points), device=device) * f(points[:, 0], points[:, 1]) for f in funcs]
    return changed_funcs
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


def get_diffusion_term_a_expr():
    return -1
    return sympy.sin(x_sym*y_sym)

def generate_u_expr(n_expr: int = 5):
    u_exprs = []
    for i in range(n_expr):
        # expr = (
        #     random.uniform(0.5, 5.0) * sympy.sin(random.randint(1, 5)*x_sym) * 
        #     sympy.cos(random.randint(1, 5)*y_sym) +
        #     random.uniform(0.1, 5.0) * x_sym**random.randint(1, 3) * y_sym**random.randint(1, 3)
        # )
        expr = (sympy.exp(-1/2*(20*(x_sym-0.5)**2 +20*(y_sym-0.5)**2))/(2*math.pi))
        u_exprs.append(expr)
    return u_exprs

def get_u_bnd_expr(n_expr: int = 5):
    u_bnd_exprs = []
    for _ in range(n_expr):
        expr = 0
        u_bnd_exprs.append(expr)
    return u_bnd_exprs

def generate_u_expr_w_bnd(domain, u_bnd_exprs):
    '''
    Generate u(x) expressions based on boundary conditions. Currently only implemented for constant boundary conditions.
    '''
    x_min, x_max, y_min, y_max = domain
    u_exprs = []
    for i in range(len(u_bnd_exprs)):
        expr = (100* ((x_sym-x_max)**random.randint(1, 2))*((x_sym-x_min)**random.randint(1, 2))
                * ((y_sym-y_max)**random.randint(1, 2))*((y_sym-y_min)**random.randint(1, 2))
                + u_bnd_exprs[i]
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
        expr = u_x * a_x + a_expr * u_xx + u_y * a_y + a_expr * u_yy
        f_exprs.append(expr)
    return f_exprs

def expr_to_func(exprs):
    funcs = [sympy.lambdify([x_sym, y_sym], expr, modules=[torch_namespace]) for expr in exprs]
    return funcs

def func_input_wrapper(funcs):
    '''
    changes functions that take x, y as input to taking points of form batch_size x 2 as input.
    '''
    changed_funcs = [lambda points, f=f: torch.ones(len(points)) * f(points[:, 0], points[:, 1]) for f in funcs]
    return changed_funcs
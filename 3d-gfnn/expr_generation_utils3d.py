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

x_sym, y_sym, z_sym = sympy.symbols('x y z')


def get_diffusion_term_a_expr_3d():
    return -1
    return sympy.sin(x_sym*y_sym*z_sym)

def generate_u_expr_3d(n_expr: int = 1):
    u_exprs = []
    for i in range(n_expr):
        # expr = (
        #     random.uniform(0.5, 5.0) * sympy.sin(random.randint(1, 5)*x_sym) * 
        #     sympy.cos(random.randint(1, 5)*y_sym) +
        #     random.uniform(0.1, 5.0) * x_sym**random.randint(1, 3) * y_sym**random.randint(1, 3)
        # )
        expr=(x_sym**2 + y_sym**2 + z_sym**2)
        u_exprs.append(expr)
    return u_exprs

def get_f_expr_3d(u_exprs, a_expr):
    f_exprs = []
    for u in u_exprs:
        u_x = sympy.diff(u, x_sym)
        u_xx = sympy.diff(u, x_sym, 2)
        u_y = sympy.diff(u, y_sym)
        u_yy = sympy.diff(u, y_sym, 2)
        u_z = sympy.diff(u, z_sym)
        u_zz = sympy.diff(u, z_sym, 2)
        a_x = sympy.diff(a_expr, x_sym)
        a_y = sympy.diff(a_expr, y_sym)
        a_z = sympy.diff(a_expr, z_sym)
        expr = u_x * a_x + a_expr * u_xx + u_y * a_y + a_expr * u_yy + u_z * a_z + a_expr * u_zz
        f_exprs.append(expr)
    return f_exprs

def expr_to_func_3d(exprs):
    funcs = [sympy.lambdify([x_sym, y_sym, z_sym], expr, modules=[torch_namespace]) for expr in exprs]
    return funcs

def func_input_wrapper_3d(funcs):
    '''
    changes functions that take x, y, z as input to taking points of form batch_size x 3 as input.
    '''
    changed_funcs = [lambda points, f=f: torch.ones(len(points)) * f(points[:, 0], points[:, 1], points[:,2]) for f in funcs]
    return changed_funcs
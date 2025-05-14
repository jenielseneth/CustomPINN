

import torch
from plot_utils import plot_multiple_points, plot_points
from data_generation_utils import generate_points
import sympy
import os

# x_sym, y_sym = sympy.symbols("x y")
# u_func = x_sym ** 2 + y_sym ** 2
# a_func = -1

# ##Calculate Source Term
# u_func_x = sympy.diff(u_func, x_sym)
# u_func_y = sympy.diff(u_func, y_sym)
# a_func_x = sympy.diff(a_func, x_sym)
# a_func_y = sympy.diff(a_func, y_sym)
# u_func_xx = sympy.diff(u_func_x, x_sym)
# u_func_yy = sympy.diff(u_func_y, y_sym)
# f_func = a_func_x*u_func_x + a_func*u_func_xx + a_func_y*u_func_y + a_func*u_func_yy
# print(type(u_func.subs([(x_sym, 0), (y_sym, 2.5)])))
# assert False

def explicit_u_func_1(points):
    '''
    points: Tensor of n x m x ... x b x 2
    '''
    x, y = points[..., 0], points[..., 1]
    return x**3 * y**3

def source_term(points):
    '''
    points: Tensor of n x m x ... x b x 2
    '''
    x, y = points[..., 0], points[..., 1]
    u_xx = 6* x * y**3
    u_yy = 6* y * x**3
    return -u_xx - u_yy

if __name__ == "__main__":  
    name = "manu_sol_2/"
    function_str = "x**3 * y**3"
    source_term_str = "-u_xx - u_yy"
    dir = "./data/" + name
    train_chebyshev = True
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        print("Warning: " + dir + " already exists.")
    x_min, x_max = 0, 1 
    y_min, y_max = 0, 1
    domain = (x_min, x_max, y_min, y_max)

    with open(dir + "info.txt", "w") as f: 
        f.write('Domain: ' + ', '.join(map(str,domain)) + "\n")
        f.write('Function: ' + function_str + "\n")
        f.write('Source Term: ' + str(train_chebyshev) + "\n")
        f.write('Train using Chebyshev points: ' + source_term_str + "\n")
    generate_points(domain=domain, num_col_points=400, eval_u_func=explicit_u_func_1, chebyshev=train_chebyshev, dir=dir, training_data=True)
    generate_points(domain=domain, num_col_points=400, num_bnd_points=80,  eval_u_func=explicit_u_func_1, chebyshev=False, dir=dir, training_data=False)


    points = torch.load(dir + "collocation_train.pt")
    coordinates = points["coordinates"]
    values = points["values"]
    plot_points(coordinates, values, title="Collocation Training Data")
    
    points = torch.load(dir + "boundary_train.pt")
    coordinates = points["coordinates"]
    values = points["values"]
    plot_points(coordinates, values,title="Boundary Training Data")
    
    points = torch.load(dir + "collocation_test.pt")
    coordinates = points["coordinates"]
    values = points["values"]
    plot_points(coordinates, values, title="Collocation Test Data")
    
    points = torch.load(dir + "boundary_test.pt")
    coordinates = points["coordinates"]
    values = points["values"]
    plot_points(coordinates, values,title="Boundary Test Data")
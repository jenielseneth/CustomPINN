

import random
import torch
from plot_utils import plot_multiple_points, plot_points
from data_generation_utils import generate_points, sample_uniform_mesh_points
import sympy
import os

def explicit_u_func_1(points):
    '''
    points: Tensor of n x m x ... x b x 2
    '''
    x, y = points[..., 0], points[..., 1]
    return x**2 + y**2

def source_term(points):
    '''
    points: Tensor of n x m x ... x b x 2
    '''
    x, y = points[..., 0], points[..., 1]
    return -6 * x * y**3 - 6 * y * x**3

if __name__ == "__main__":  
    name = "manu_sol_2/"
    function_str = "x**3 * y**3"
    source_term_str = "-6 * x * y**3 - 6 * y * x**3"
    dir = "./data/" + name
    train_chebyshev = True
    uniform_quadrature = True
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
        f.write('Train using Chebyshev points: ' + source_term_str + "\n")
    
    qudrature_num_points = (20, 20)
    generate_points(domain=domain, num_points=(20, 20), qudrature_num_points=qudrature_num_points, u_gt_func=explicit_u_func_1, f_func=source_term, u_gt_expr=function_str, f_func_expr=source_term_str, chebyshev=True, dir=dir, training_data=True)
    generate_points(domain=domain, num_points=(400,), qudrature_num_points=qudrature_num_points, u_gt_func=explicit_u_func_1, f_func=source_term, u_gt_expr=function_str, f_func_expr=source_term_str, chebyshev=False, dir=dir, training_data=False)

    print(dir + "data_train.pt")
    points = torch.load(dir + "data_train.pt")
    coordinates = points["coordinates"]
    values = points["u_values"]
    plot_points(coordinates, values, title="Training u(x) Data")
    
    values = points["f_values"]
    uniform_points = points["f_mesh"]
    plot_points(uniform_points, values,title="Training f(x) Data")
    
    points = torch.load(dir + "data_test.pt")
    coordinates = points["coordinates"]
    values = points["u_values"]
    plot_points(coordinates, values, title="Test u(x) Data")
    
    values = points["f_values"]
    uniform_points = points["f_mesh"]
    plot_points(uniform_points, values,title="Test f(x) Data")
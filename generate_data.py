

import torch
from chebyshev import plot_multiple_points, plot_points
from data_generation_utils import generate_points
from pde_utils import get_u_evaluation_func, greens_function_poisson_eq_2d
import logging
import os


def explicit_u_func_1(point, domain):
    x, y = point
    return 1 + x**2 + 2 * y**2

def source_term(x,y):
    return -6

if __name__ == "__main__":  
    name = "manu_sol_1/"
    function_str = "1 + x**2 + 2 * y**2"
    source_term_str = "-6"
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
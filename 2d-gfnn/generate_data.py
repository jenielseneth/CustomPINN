

from datetime import datetime
import random
import typing
import torch
from plot_utils import plot_multiple_points, plot_points
from expr_generation_utils import expr_to_func, func_input_wrapper, generate_u_expr, generate_u_expr_w_bnd, get_diffusion_term_a_expr, get_f_expr, get_u_bnd_expr
from data_generation_utils import generate_points, sample_points 
import math
import os
from constants_utils import mesh_type

def generate_data(u_expressions: list, u_bnd_expression, f_expressions: list, a_expression, 
            main_save_dir: str, domain: tuple, mesh_size: tuple, train_data: bool,
            u_mesh_type: mesh_type = "chebyshev",
            f_mesh_type: mesh_type = "chebyshev",
            f_mesh_num_points: tuple = None):
    
    #Default to using chebyshev points for f_mesh and values.
    if f_mesh_num_points is None:
        f_mesh_num_points = mesh_size

    #Get from the expressions the appropriate functions.
    u_funcs = expr_to_func(u_expressions)
    f_funcs = expr_to_func(f_expressions)
    u_changed = func_input_wrapper(u_funcs)
    f_changed = func_input_wrapper(f_funcs)

    function_strs = [str(u_expr) for u_expr in u_expressions]
    u_bnd_str = str(u_bnd_expression)
    source_term_strs = [str(f_expression) for f_expression in f_expressions]

    generate_points(domain=domain, u_mesh_num_points=mesh_size, 
                u_gt_funcs=u_changed, f_funcs=f_changed,
                u_mesh_type=u_mesh_type, f_mesh_type=f_mesh_type,
                save_dir=main_save_dir, training_data=train_data, 
                f_mesh_num_points=f_mesh_num_points, 
                u_gt_exprs=function_strs, f_func_exprs=source_term_strs, u_bnd_expr=u_bnd_str)


    info_file_name = "train_" + "info.txt" if train_data else "test_" + "info.txt"
    fncs_file_name = "train_fncs.txt" if train_data else "test_fncs.txt"
    #Individual data information
    with open(main_save_dir + info_file_name, "w") as f: 
        f.write('Domain: ' + ', '.join(map(str,domain)) + "\n")
        f.write('Boundary term for u(x): ' + u_bnd_str + "\n")
        f.write('Diffusion Term a(x): ' + str(a_expression) + "\n")
        f.write('u(x) Mesh Size: ' + ', '.join(map(str,mesh_size)) + "\n")
        f.write('u(x) Mesh Type: ' + str(u_mesh_type) + "\n")
        f.write('f(x) Mesh Type: ' + str(f_mesh_type)+ "\n")
        f.write('f(x) Mesh Size: ' + ', '.join(map(str,f_mesh_num_points))+ "\n")
    
    #Iterate for all u_expressions
    with open(main_save_dir + fncs_file_name, "w") as f:
        for i, _ in enumerate(u_expressions):
            f.write(str(i+1) + '. GT u(x): ' + function_strs[i] + "; Source Term f(x): " + source_term_strs[i] + "\n")

    name = "data_train.pt" if train_data else "data_test.pt"
    plot_title = "Training " if train_data else "Test "
    points = torch.load(main_save_dir + name)
    test_ind = 0
    slice_ind = slice(*points["data_addresses"][test_ind])
    u_coordinates = points["coordinates"][slice_ind]
    u_values = points["u_values"][slice_ind]
    plot_points(u_coordinates, u_values, title=plot_title + f'u(x) Data')
    
    f_values = points["f_values"][test_ind]
    f_mesh = points["f_meshes"][test_ind]
    plot_points(f_mesh, f_values,title=plot_title + f'f(x) Data')


if __name__ == "__main__":  

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    ##Define domain values
    x_min, x_max = 0, 1 
    y_min, y_max = 0, 1
    domain = (x_min, x_max, y_min, y_max)

    ##Generate source terms, u_functions and diffusion term
    boundary = True
    n_u_expr = 10

    if boundary == True:
        u_bnd_expr = get_u_bnd_expr()
    else: 
        u_bnd_expr = None

    a_expr = get_diffusion_term_a_expr()
    u_train_exprs = generate_u_expr(n_expr=n_u_expr) if not boundary else generate_u_expr_w_bnd(domain=domain, u_bnd_expr=u_bnd_expr, n_expr=n_u_expr)
    f_train_exprs = get_f_expr(u_train_exprs, a_expr)
    u_test_exprs = generate_u_expr(n_expr=n_u_expr) if not boundary else generate_u_expr_w_bnd(domain=domain, u_bnd_expr=u_bnd_expr, n_expr=n_u_expr)
    f_test_exprs = get_f_expr(u_test_exprs, a_expr)

    u_train_mesh_type: mesh_type = "chebyshev"
    u_train_mesh_size = (20,20)
    u_test_mesh_type: mesh_type = "chebyshev"
    u_test_mesh_size = (20,20)
    f_mesh_type: mesh_type = "chebyshev"
    f_mesh_size = (31, 31)

    #Ensure the chebyshev points don't overlap:
    overlap = False
    if not math.gcd(u_train_mesh_size[0], f_mesh_size[0]) == 1:
        print(f"Warning: Chebyshev points for u_train and f mesh sizes in dim 0 have gcd larger than 1: {math.gcd(u_train_mesh_size[0], f_mesh_size[0])}, they may overlap.")
        overlap = True
    if not math.gcd(u_train_mesh_size[1], f_mesh_size[1]) == 1:
        print(f"Warning: Chebyshev points for u_train and f mesh sizes in dim 1 have gcd larger than 1: {math.gcd(u_train_mesh_size[1], f_mesh_size[1])}, they may overlap.")
        overlap = True
    if not math.gcd(u_test_mesh_size[0], f_mesh_size[0]) == 1:
        print(f"Warning: Chebyshev points for u_test and f mesh sizes in dim 0 have gcd larger than 1: {math.gcd(u_test_mesh_size[0], f_mesh_size[0])}, they may overlap.")
        overlap = True
    if not math.gcd(u_test_mesh_size[1], f_mesh_size[1]) == 1:
        print(f"Warning: Chebyshev points for u_test and f mesh sizes in dim 1 have gcd larger than 1: {math.gcd(u_test_mesh_size[1], f_mesh_size[1])}, they may overlap.")
        overlap = True
    if overlap:
        user_input = input("The chebyshev points may overlap, do you want to continue? (y/n): ")
        if user_input.lower() != 'y':
            print("Exiting data generation.")
            exit()
    else:
        print("Chebyshev points do not overlap, proceeding with data generation.")
    
        

    plot_uniform_mesh = sample_points(domain=domain, mesh_size=(20, 20), mesh_type="uniform")
    a_func_values = func_input_wrapper(expr_to_func([a_expr]))[0](plot_uniform_mesh)
    plot_points(plot_uniform_mesh, a_func_values, title="Diffusion term values")

    dir = f"./res/{timestamp}/data/" #Main directory

    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        print("Warning: " + dir + " already exists.")

    # train_dir = dir + "train/" #Train directory
    # test_dir = dir + "test/" #Test directory

    #Train data
    generate_data(u_expressions=u_train_exprs, u_bnd_expression=u_bnd_expr, f_expressions=f_train_exprs,
                  a_expression=a_expr, main_save_dir=dir, domain=domain, mesh_size=u_train_mesh_size,
                  train_data=True, u_mesh_type=u_train_mesh_type, f_mesh_type=f_mesh_type, f_mesh_num_points=f_mesh_size)

    #Test data
    generate_data(u_expressions=u_test_exprs, u_bnd_expression=u_bnd_expr, f_expressions=f_test_exprs,
                  a_expression=a_expr, main_save_dir=dir, domain=domain, mesh_size=u_test_mesh_size,
                  train_data=False, u_mesh_type=u_test_mesh_type, f_mesh_type=f_mesh_type, f_mesh_num_points=f_mesh_size)



from datetime import datetime
import random
import torch
from plot_utils import plot_multiple_points, plot_points
from expr_generation_utils import expr_to_func, func_input_wrapper, generate_u_expr, generate_u_expr_w_bnd, get_diffusion_term_a_expr, get_f_expr, get_u_bnd_expr
from data_generation_utils import generate_points, sample_uniform_mesh_points 
import sympy
import os

def generate_data(u_expressions: list, u_bnd_expressions: list, f_expressions: list, a_expression, 
             main_save_dir: str, domain: tuple, mesh_size: tuple, train_data: bool, u_chebyshev_mesh: bool,
             uniform_f_quadrature: bool = False, f_mesh_num_points: tuple = None):
    
    #Default to using chebyshev points for f_mesh and values.
    if f_mesh_num_points is None:
        f_mesh_num_points = mesh_size

    #Get from the expressions the appropriate functions.
    u_funcs = expr_to_func(u_expressions)
    f_funcs = expr_to_func(f_expressions)
    u_changed = func_input_wrapper(u_funcs)
    f_changed = func_input_wrapper(f_funcs)

    #Iterate for all u_expressions
    for i, u_expr in enumerate(u_expressions):

        ##Log Information
        function_str = str(u_expr)
        u_bnd_str = str(u_bnd_expressions[i])
        source_term_str = str(f_expressions[i])
        subdir = main_save_dir + str(i) + "/"
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        else:
            print("Warning: " + subdir + " already exists.")
        #Individual data information
        with open(subdir + "info.txt", "w") as f: 
            f.write('Domain: ' + ', '.join(map(str,domain)) + "\n")
            f.write('GT u(x): ' + function_str + "\n")
            f.write('Boundary term for u(x): ' + u_bnd_str + "\n")
            f.write('Diffusion Term a(x): ' + str(a_expression) + "\n")
            f.write('Source Term f(x): ' + source_term_str + "\n")
            f.write('Mesh size: ' + ', '.join(map(str,mesh_size)) + "\n")
            f.write('Mesh uses Chebyshev points: ' + str(u_chebyshev_mesh) + "\n")
            f.write('Uniform Mesh for source term: ' + str(uniform_f_quadrature)+ "\n")
            f.write('Source Term mesh size: ' + ', '.join(map(str,f_mesh_num_points))+ "\n")
        
        ##Generate Points
        generate_points(domain=domain, num_points=mesh_size, 
                        u_gt_func=u_changed[i], f_func=f_changed[i],
                        u_gt_expr=function_str, f_func_expr=source_term_str, 
                        u_chebyshev_mesh=u_chebyshev_mesh, save_dir=subdir, training_data=train_data, 
                        uniform_f_quadrature=uniform_f_quadrature, f_qudrature_num_points=f_mesh_num_points)
        
        name = "data_train.pt" if train_data else "data_test.pt"
        plot_title = "Training " if train_data else "Test "
        points = torch.load(subdir + name)
        u_coordinates = points["coordinates"]
        u_values = points["u_values"]
        plot_points(u_coordinates, u_values, title=plot_title + f'u_{i}(x) Data')
        
        f_values = points["f_values"]
        f_mesh = points["f_mesh"]
        plot_points(f_mesh, f_values,title=plot_title + f'f_{i}(x) Data')


if __name__ == "__main__":  

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    ##Define domain values
    x_min, x_max = 0, 1 
    y_min, y_max = 0, 1
    domain = (x_min, x_max, y_min, y_max)

    ##Generate source terms, u_functions and diffusion term
    boundary = True
    n_u_expr = 5

    if boundary == True:
        u_bnd_exprs = get_u_bnd_expr(n_expr=n_u_expr)
        u_bnd_funcs = expr_to_func(u_bnd_exprs)
        u_bnd_changed = func_input_wrapper(u_bnd_funcs)
    else: 
        u_bnd_changed = u_bnd_exprs = [None for _ in range(n_u_expr)]

    u_exprs = generate_u_expr(n_expr=n_u_expr) if not boundary else generate_u_expr_w_bnd(domain=domain, u_bnd_exprs=u_bnd_exprs)
    a_expr = get_diffusion_term_a_expr()
    f_exprs = get_f_expr(u_exprs, a_expr)
    print("u expr: ", u_exprs)
    print("a expr: ", a_expr)
    print("f expr: ", f_exprs)

    training_mesh_size = (20, 20)
    test_points_size = (400,)
    qudrature_num_points = (20, 20)

    train_chebyshev = True  ##Generate Training data using Chebyshev nodes - Test data is generated using random points.
    uniform_f_quadrature = False  ##Generate Source term data using a uniform mesh.
    f_mesh_points = None

    plot_uniform_mesh = sample_uniform_mesh_points((0, 1, 0, 1), (20, 20))
    a_func_values = func_input_wrapper(expr_to_func([a_expr]))[0](plot_uniform_mesh)
    plot_points(plot_uniform_mesh, a_func_values, title="Diffusion term values")

    dir = f"./res/{timestamp}/data/" #Main directory

    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        print("Warning: " + dir + " already exists.")

    train_dir = dir + "train/" #Train directory
    test_dir = dir + "test/" #Test directory

    #Train data
    generate_data(u_expressions=u_exprs, u_bnd_expressions=u_bnd_exprs, f_expressions=f_exprs,
                  a_expression=a_expr, main_save_dir=train_dir, domain=domain, mesh_size=training_mesh_size,
                  train_data=True, u_chebyshev_mesh=True, 
                  uniform_f_quadrature=uniform_f_quadrature, f_mesh_num_points=f_mesh_points)

    #Test data
    generate_data(u_expressions=u_exprs, u_bnd_expressions=u_bnd_exprs, f_expressions=f_exprs,
                  a_expression=a_expr, main_save_dir=test_dir, domain=domain, mesh_size=training_mesh_size,
                  train_data=False, u_chebyshev_mesh=True, 
                  uniform_f_quadrature=uniform_f_quadrature, f_mesh_num_points=f_mesh_points)

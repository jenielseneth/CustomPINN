

import random
import torch
from plot_utils3d import plot_multiple_points_3d, plot_points_3d
from expr_generation_utils3d import expr_to_func_3d, func_input_wrapper_3d, generate_u_expr_3d, get_diffusion_term_a_expr_3d, get_f_expr_3d
from data_generation_utils3d import generate_points_3d, sample_uniform_mesh_points_3d 
import sympy
import os

if __name__ == "__main__":  

    ##Generate source terms, u_functions and diffusion term
    u_exprs = generate_u_expr_3d(n_expr=1)
    a_expr = get_diffusion_term_a_expr_3d()
    f_exprs = get_f_expr_3d(u_exprs, a_expr)
    u_funcs = expr_to_func_3d(u_exprs)
    f_funcs = expr_to_func_3d(f_exprs)
    print("u expr: ", u_exprs)
    print("a expr: ", a_expr)
    print("f expr: ", f_exprs)
    u_changed = func_input_wrapper_3d(u_funcs)
    f_changed = func_input_wrapper_3d(f_funcs)

    x_min, x_max = 0, 1 
    y_min, y_max = 0, 1
    z_min, z_max = 0, 1
    domain = (x_min, x_max, y_min, y_max, z_min, z_max)

    training_mesh_size = (10, 10, 10)
    test_points_size = (400,)
    qudrature_num_points = (10, 10, 10)

    train_chebyshev = True  ##Generate Training data using Chebyshev nodes - Test data is generated using random points.
    uniform_quadrature = True  ##Generate Source term data using a uniform mesh.

    plot_uniform_mesh = sample_uniform_mesh_points_3d((0, 5, 0, 5, 0, 5), (10, 10, 10))
    a_func_values = func_input_wrapper_3d(expr_to_func_3d([a_expr]))[0](plot_uniform_mesh)
    print(a_func_values)
    plot_points_3d(plot_uniform_mesh, a_func_values, title="Diffusion term values")

    name = str(a_expr) + "/"
    dir = "./res/" + name + "data/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        print("Warning: " + dir + " already exists.")

    for i, u_expr in enumerate(u_exprs):

        ##Log Information
        function_str = str(u_expr)
        source_term_str = str(f_exprs[i])
        print(name, function_str, source_term_str)
        subdir = dir + str(i) + "/"
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        else:
            print("Warning: " + subdir + " already exists.")

        with open(subdir + "info.txt", "w") as f: 
            f.write('Domain: ' + ', '.join(map(str,domain)) + "\n")
            f.write('GT u(x): ' + function_str + "\n")
            f.write('Diffusion Term a(x): ' + str(a_expr))
            f.write('Source Term f(x): ' + source_term_str + "\n")
            f.write('Train mesh size: ' + ', '.join(map(str,training_mesh_size)) + "\n")
            f.write('Test mesh size: ' + ', '.join(map(str,test_points_size)) + "\n")
            f.write('Train using Chebyshev points: ' + str(train_chebyshev) + "\n")
            f.write('Uniform Mesh for source term: ' + str(uniform_quadrature)+ "\n")
            f.write('Source Term mesh size: ' + ', '.join(map(str,qudrature_num_points))+ "\n")
        
        ##Generate Points
        generate_points_3d(domain=domain, num_points=training_mesh_size, qudrature_num_points=qudrature_num_points, u_gt=u_changed[i], f_func=f_changed[i], u_gt_expr=function_str, f_func_expr=source_term_str, chebyshev=True, dir=subdir, training_data=True)
        generate_points_3d(domain=domain, num_points=test_points_size, qudrature_num_points=qudrature_num_points, u_gt=u_changed[i], f_func=f_changed[i], u_gt_expr=function_str, f_func_expr=source_term_str, chebyshev=False, dir=subdir, training_data=False)

        print(subdir + "data_train.pt")
        points = torch.load(subdir + "data_train.pt")
        coordinates = points["coordinates"]
        values = points["u_values"]
        plot_points_3d(coordinates, values, title=f'Training u_{i}(x,y,z) Data')
        
        values = points["f_values"]
        uniform_points = points["f_mesh"]
        plot_points_3d(uniform_points, values,title=f'Training f_{i}(x,y,z) Data')
        
        points = torch.load(subdir + "data_test.pt")
        coordinates = points["coordinates"]
        values = points["u_values"]
        plot_points_3d(coordinates, values, title=f'Test u_{i}(x,y,z) Data')
        
        values = points["f_values"]
        uniform_points = points["f_mesh"]
        plot_points_3d(uniform_points, values,title=f'Test f_{i}(x,y,z) Data')
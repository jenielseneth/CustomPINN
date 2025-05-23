

from datetime import datetime
import random
import torch
from plot_utils import plot_multiple_points, plot_points
from expr_generation_utils import expr_to_func, func_input_wrapper, generate_u_expr, generate_u_expr_w_bnd, get_diffusion_term_a_expr, get_f_expr, get_u_bnd_expr
from data_generation_utils import generate_points, sample_uniform_mesh_points 
import sympy
import os

if __name__ == "__main__":  

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    ##Define domain values
    x_min, x_max = 0, 1 
    y_min, y_max = 0, 1
    domain = (x_min, x_max, y_min, y_max)

    ##Generate source terms, u_functions and diffusion term
    boundary = False
    n_u_expr = 1

    if boundary == True:
        u_bnd_exprs = get_u_bnd_expr(n_expr=n_u_expr)
        u_bnd_funcs = expr_to_func(u_bnd_exprs)
        u_bnd_changed = func_input_wrapper(u_bnd_funcs)
    else: 
        u_bnd_changed = u_bnd_exprs = [None for _ in range(n_u_expr)]

    u_exprs = generate_u_expr(n_expr=n_u_expr) if not boundary else generate_u_expr_w_bnd(domain=domain, u_bnd_exprs=u_bnd_exprs)
    a_expr = get_diffusion_term_a_expr()
    f_exprs = get_f_expr(u_exprs, a_expr)
    u_funcs = expr_to_func(u_exprs)
    f_funcs = expr_to_func(f_exprs)
    print("u expr: ", u_exprs)
    print("a expr: ", a_expr)
    print("f expr: ", f_exprs)
    u_changed = func_input_wrapper(u_funcs)
    f_changed = func_input_wrapper(f_funcs)

    training_mesh_size = (20, 20)
    test_points_size = (400,)
    qudrature_num_points = (20, 20)

    train_chebyshev = True  ##Generate Training data using Chebyshev nodes - Test data is generated using random points.
    uniform_quadrature = True  ##Generate Source term data using a uniform mesh.

    plot_uniform_mesh = sample_uniform_mesh_points((0, 1, 0, 1), (20, 20))
    a_func_values = func_input_wrapper(expr_to_func([a_expr]))[0](plot_uniform_mesh)
    plot_points(plot_uniform_mesh, a_func_values, title="Diffusion term values")

    # name = str(a_expr) 
    dir = f"./res/{timestamp}/data/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        print("Warning: " + dir + " already exists.")

    for i, u_expr in enumerate(u_exprs):

        ##Log Information
        function_str = str(u_expr)
        u_bnd_str = str(u_bnd_exprs[i])
        source_term_str = str(f_exprs[i])
        subdir = dir + str(i) + "/"
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        else:
            print("Warning: " + subdir + " already exists.")

        with open(subdir + "info.txt", "w") as f: 
            f.write('Domain: ' + ', '.join(map(str,domain)) + "\n")
            f.write('GT u(x): ' + function_str + "\n")
            f.write('Boundary term for u(x): ' + u_bnd_str + "\n")
            f.write('Diffusion Term a(x): ' + str(a_expr) + "\n")
            f.write('Source Term f(x): ' + source_term_str + "\n")
            f.write('Train mesh size: ' + ', '.join(map(str,training_mesh_size)) + "\n")
            f.write('Test mesh size: ' + ', '.join(map(str,test_points_size)) + "\n")
            f.write('Train using Chebyshev points: ' + str(train_chebyshev) + "\n")
            f.write('Uniform Mesh for source term: ' + str(uniform_quadrature)+ "\n")
            f.write('Source Term mesh size: ' + ', '.join(map(str,qudrature_num_points))+ "\n")
        
        ##Generate Points
        generate_points(domain=domain, num_points=training_mesh_size, qudrature_num_points=qudrature_num_points, 
                        u_gt_func=u_changed[i], f_func=f_changed[i], u_bnd_func=u_bnd_changed[i], 
                        u_gt_expr=function_str, f_func_expr=source_term_str, 
                        chebyshev=True, dir=subdir, training_data=True)
        generate_points(domain=domain, num_points=(400,), qudrature_num_points=qudrature_num_points, 
                        u_gt_func=u_changed[i], f_func=f_changed[i], u_bnd_func=u_bnd_changed[i],
                        u_gt_expr=function_str, f_func_expr=source_term_str, 
                        chebyshev=False, dir=subdir, training_data=False)

        print(subdir + "data_train.pt")
        points = torch.load(subdir + "data_train.pt")
        coordinates = points["coordinates"]
        values = points["u_values"]
        plot_points(coordinates, values, title=f'Training u_{i}(x) Data')
        
        values = points["f_values"]
        uniform_points = points["f_mesh"]
        plot_points(uniform_points, values,title=f'Training f_{i}(x) Data')
        
        points = torch.load(subdir + "data_test.pt")
        coordinates = points["coordinates"]
        values = points["u_values"]
        plot_points(coordinates, values, title=f'Test u_{i}(x) Data')
        
        values = points["f_values"]
        uniform_points = points["f_mesh"]
        plot_points(uniform_points, values,title=f'Test f_{i}(x) Data')
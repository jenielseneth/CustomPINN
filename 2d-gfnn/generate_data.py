

from datetime import datetime
import random
import typing
import torch
from plot_utils import plot_multiple_points, plot_points
from expr_generation_utils import expr_to_func, func_input_wrapper, generate_u_expr, generate_u_expr_w_bnd, get_diffusion_term_a_expr, get_f_expr, get_u_bnd_expr
from data_generation_utils import generate_points, sample_points 
from constants_utils import DataGenerationParameters
from random_utils import log_dict_as_json, retrieve_dict_from_json
import math
import os
from constants_utils import mesh_type

if __name__ == "__main__":  

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    ##Define domain values
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    domain = (x_min, x_max, y_min, y_max)
    
    # Ask about pre-generated function expressions.
    user_input = input("Model dir of existing expressions? Press enter to skip:")

    if os.path.exists("./res/" + user_input) and user_input != "":
        print("Using pre-existing expressions.")
        # Use pre-generated function expressions
        data_dir = "./res/" + user_input + "/data/"
        train_params = DataGenerationParameters(**retrieve_dict_from_json(data_dir+"train_params.json")).str_to_sympy_expr()
        test_params = DataGenerationParameters(**retrieve_dict_from_json(data_dir+"test_params.json")).str_to_sympy_expr()
        u_bnd_expr = train_params.u_bnd_expr
        a_expr = train_params.a_diffusion_expr
        u_train_exprs = train_params.u_func_exprs
        f_train_exprs = train_params.f_func_exprs
        u_test_exprs = test_params.u_func_exprs
        f_test_exprs = test_params.f_func_exprs

        domain = train_params.domain
        
    elif user_input == "":
        print("Generating function expressions.")
        # Generate source terms, u_functions and diffusion term
        boundary = True
        n_u_expr = 100

        if boundary == True:
            u_bnd_expr = get_u_bnd_expr()
        else: 
            u_bnd_expr = None

        a_expr = get_diffusion_term_a_expr()
        u_train_exprs = generate_u_expr(n_expr=n_u_expr) if not boundary else generate_u_expr_w_bnd(domain=domain, u_bnd_expr=u_bnd_expr, n_expr=n_u_expr)
        f_train_exprs = get_f_expr(u_train_exprs, a_expr)
        u_test_exprs = generate_u_expr(n_expr=n_u_expr) if not boundary else generate_u_expr_w_bnd(domain=domain, u_bnd_expr=u_bnd_expr, n_expr=n_u_expr)
        f_test_exprs = get_f_expr(u_test_exprs, a_expr)
    else:
        print("Please provide a valid directory or press enter.")
        raise NotADirectoryError(f"{"./res/" + user_input} is not a valid directory.")
    u_train_mesh_type: mesh_type = "chebyshev"
    u_train_mesh_size = (20,20)
    u_test_mesh_type: mesh_type = "chebyshev"
    u_test_mesh_size = (20,20)
    f_mesh_type: mesh_type = "chebyshev"
    f_mesh_sizes = [(5,5), (10,10), (15,15), (21,21), (25, 25), (27, 27), (30,30), (35, 35)]
    # f_mesh_sizes = [ (3,2), (4,3), (5,4), (6, 5), (7, 6)]
    # f_mesh_sizes = [(10,10)]
    f_mesh_sizes = [(3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (10,10), (15,15)]

    for i, f_mesh_size in enumerate(f_mesh_sizes):
        #Ensure the chebyshev points don't overlap:
        overlap = False

        if u_test_mesh_type == f_mesh_type == "chebyshev":
            #check size-1 for chebyshev, see data_generation_utils: _sample_chebyshev_points
            if not math.gcd(u_train_mesh_size[0]-1, f_mesh_size[0]-1) == 1:
                print(f"Warning: Chebyshev points sizes for u_train {u_train_mesh_size} and f mesh {f_mesh_size} in dim 0 have gcd larger than 1: {math.gcd(u_train_mesh_size[0]-1, f_mesh_size[0]-1)}, they may overlap.")
                overlap = True
            if not math.gcd(u_train_mesh_size[1]-1, f_mesh_size[1]-1) == 1:
                print(f"Warning: Chebyshev points sizes for u_train {u_train_mesh_size} and f mesh {f_mesh_size} in dim 1 have gcd larger than 1: {math.gcd(u_train_mesh_size[1]-1, f_mesh_size[1]-1)}, they may overlap.")
                overlap = True
            if not math.gcd(u_test_mesh_size[0]-1, f_mesh_size[0]-1) == 1:
                print(f"Warning: Chebyshev points sizes for u_train {u_train_mesh_size} and f mesh {f_mesh_size} in dim 0 have gcd larger than 1: {math.gcd(u_test_mesh_size[0]-1, f_mesh_size[0]-1)}, they may overlap.")
                overlap = True
            if not math.gcd(u_test_mesh_size[1]-1, f_mesh_size[1]-1) == 1:
                print(f"Warning: Chebyshev points sizes for u_train {u_train_mesh_size} and f mesh {f_mesh_size} in dim 1 have gcd larger than 1: {math.gcd(u_test_mesh_size[1]-1, f_mesh_size[1]-1)}, they may overlap.")
                overlap = True
            if overlap:
                user_input = input("The chebyshev points may overlap, do you want to continue? (y/n): ")
                if user_input.lower() != 'y':
                    print(f"Skipping data generation with f mesh {f_mesh_size}.")
                    continue
            else:
                print("Chebyshev points do not overlap, proceeding with data generation.")
        
            
        plot_uniform_mesh = sample_points(domain=domain, mesh_size=(20, 20), mesh_type="uniform")
        a_func_values = func_input_wrapper(expr_to_func([a_expr]))[0](plot_uniform_mesh)
        plot_points(plot_uniform_mesh, a_func_values, title="Diffusion term values")

        dir = f"./res/{timestamp}{i}/data/" #Main directory

        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            print("Warning: " + dir + " already exists.")
        
        # #Train data 
        # generate_points(domain=domain,     
        #             u_exprs=u_train_exprs, f_exprs=f_train_exprs,
        #             u_mesh_size=u_train_mesh_size, 
        #             u_mesh_type=u_train_mesh_type, f_mesh_type=f_mesh_type,
        #             save_dir=dir, training_data=True, 
        #             f_mesh_size=f_mesh_size, 
        #             a_expression=a_expr, u_bnd_expr=u_bnd_expr)
        
        # #Test data 
        # generate_points(domain=domain,     
        #             u_exprs=u_test_exprs, f_exprs=f_test_exprs,
        #             u_mesh_size=u_test_mesh_size, 
        #             u_mesh_type=u_test_mesh_type, f_mesh_type=f_mesh_type,
        #             save_dir=dir, training_data=False, 
        #             f_mesh_size=f_mesh_size, 
        #             a_expression=a_expr, u_bnd_expr=u_bnd_expr)



        # Train data
        generate_points(domain=domain, save_dir=dir, 
                        file_name="data_train.pt", log_file_name="train_params.json",
                        num_f_terms=n_u_expr, 
                        u_mesh_size=u_train_mesh_size, u_mesh_type=u_train_mesh_type, 
                        f_mesh_size=f_mesh_size, f_mesh_type=f_mesh_type)
        

        # Test data
        generate_points(domain=domain, save_dir=dir, 
                        file_name="data_test.pt", log_file_name="test_params.json",
                        num_f_terms=n_u_expr, 
                        u_mesh_size=u_train_mesh_size, u_mesh_type=u_train_mesh_type, 
                        f_mesh_size=f_mesh_size, f_mesh_type=f_mesh_type)




        ##Test if data generation went well
        name = "data_train.pt"
        plot_title = "Training "
        points = torch.load(dir + name)
        train_ind = random.randint(0, len(u_train_exprs)-1)
        slice_ind = slice(*points["data_addresses"][train_ind])
        u_coordinates = points["coordinates"][slice_ind]
        u_values = points["u_values"][slice_ind]
        plot_points(u_coordinates, u_values, title=plot_title + f'u(x) Data')
        
        f_values = points["f_values"][train_ind]
        f_mesh = points["f_meshes"][train_ind]
        plot_points(f_mesh, f_values,title=plot_title + f'f(x) Data')


        name = "data_test.pt"
        plot_title = "Test "
        points = torch.load(dir + name)
        test_ind = random.randint(0, len(u_test_exprs)-1)
        slice_ind = slice(*points["data_addresses"][test_ind])
        u_coordinates = points["coordinates"][slice_ind]
        u_values = points["u_values"][slice_ind]
        plot_points(u_coordinates, u_values, title=plot_title + f'u(x) Data')
        
        f_values = points["f_values"][test_ind]
        f_mesh = points["f_meshes"][test_ind]
        plot_points(f_mesh, f_values,title=plot_title + f'f(x) Data')
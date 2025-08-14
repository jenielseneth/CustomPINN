

from datetime import datetime
import random
import typing
import torch
from plot_utils import plot_multiple_points, plot_points
from expr_generation_utils import expr_to_func, func_input_wrapper, generate_u_expr, generate_u_expr_w_bnd, get_diffusion_term_a_expr, get_f_expr, get_u_bnd_expr
from data_generation_utils import generate_points, multiple_f_meshes_generate_points, sample_points 
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
        train_params = DataGenerationParameters(**retrieve_dict_from_json(data_dir+"train_params.json"))
        test_params = DataGenerationParameters(**retrieve_dict_from_json(data_dir+"test_params.json"))

        domain = train_params.domain
        
    elif user_input == "":
        print("Generating function expressions.")
        darcy_flow = False

        if darcy_flow:
            assert False, "Darcy flow not implemented yet, haven't found a workaround for saving darcy flow points for multiple sized u_meshes."

        n_u_train = 100
        n_u_test = 100
        u_train_mesh_type: mesh_type = "chebyshev"
        u_train_mesh_sizes = [(8, 8), (12, 12), (20, 20)]
        u_test_mesh_type: mesh_type = "chebyshev"
        u_test_mesh_sizes = [(8, 8), (12, 12), (20, 20)]
        f_mesh_type: mesh_type = "chebyshev"
        f_mesh_sizes = [(4,4), (6,6), (9,9), (13,13), (16,16), (18,18), (21,21)]
        # f_mesh_sizes = [(21,21)]
        mesh_size_tuples = []


        dir = f"./res/{timestamp}/data/" #Main directory

        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            print("Warning: " + dir + " already exists.")
        
        # Train data
        multiple_f_meshes_generate_points(
                        domain=domain, save_dir=dir, 
                        file_name="data_train.pt", log_file_name="train_params.json",
                        num_f_terms=n_u_train, 
                        u_mesh_sizes=u_train_mesh_sizes, u_mesh_type=u_train_mesh_type, 
                        f_mesh_sizes=f_mesh_sizes, f_mesh_type=f_mesh_type, darcy_flow=darcy_flow)
        
        # Get diffusion parameters
        if darcy_flow:
            train_params = DataGenerationParameters(**retrieve_dict_from_json(dir+"train_params.json"))

        # Test data
        multiple_f_meshes_generate_points(
                        domain=domain, save_dir=dir, 
                        file_name="data_test.pt", log_file_name="test_params.json",
                        num_f_terms=n_u_test, 
                        u_mesh_sizes=u_test_mesh_sizes, u_mesh_type=u_train_mesh_type, 
                        f_mesh_sizes=f_mesh_sizes, f_mesh_type=f_mesh_type, darcy_flow=darcy_flow,
                        diffusion_gaussian_parameters=train_params.diffusion_params if darcy_flow else None
                        )

        ##Test if data generation went well
        #Check Training data
        name = "data_train.pt"
        points = torch.load(dir + name)
        train_ind = random.randint(0, n_u_train-1)
        u_slice_ind = slice(*points["u_data_addresses"][train_ind])
        f_slice_ind = slice(*points["f_data_addresses"][train_ind])
        #Check train diffusion term
        if darcy_flow:
            train_params = DataGenerationParameters(**retrieve_dict_from_json(dir+"train_params.json"))
            plot_title = "Train Diffusion Term a(x)"
            u_coordinates = points["coordinates"][u_slice_ind]
            diffusion_eval_values = points["diffusion_eval_point_values"]
            plot_points(u_coordinates, diffusion_eval_values, title=plot_title)

        #Check random source term
        plot_title = "Training "
        u_coordinates = points["coordinates"][u_slice_ind]
        u_values = points["u_values"][u_slice_ind]
        plot_points(u_coordinates, u_values, title=plot_title + f'u(x) Data')
        
        f_values = points["f_values"][f_slice_ind]
        f_mesh = points["f_meshes"][f_slice_ind]
        plot_points(f_mesh, f_values,title=plot_title + f'f(x) Data')

        #Check Test data
        name = "data_test.pt"
        points = torch.load(dir + name)
        test_ind = random.randint(0, n_u_test-1)
        u_slice_ind = slice(*points["u_data_addresses"][test_ind])
        f_slice_ind = slice(*points["f_data_addresses"][test_ind])
        #Check test diffusion term
        if darcy_flow:
            train_params = DataGenerationParameters(**retrieve_dict_from_json(dir+"test_params.json"))
            plot_title = "Test Diffusion Term a(x)"
            u_coordinates = points["coordinates"][u_slice_ind]
            diffusion_eval_values = points["diffusion_eval_point_values"]
            plot_points(u_coordinates, diffusion_eval_values, title=plot_title)


        #Check random source term
        plot_title = "Test "
        u_coordinates = points["coordinates"][u_slice_ind]
        u_values = points["u_values"][u_slice_ind]
        plot_points(u_coordinates, u_values, title=plot_title + f'u(x) Data')
        
        f_values = points["f_values"][f_slice_ind]
        f_mesh = points["f_meshes"][f_slice_ind]
        plot_points(f_mesh, f_values,title=plot_title + f'f(x) Data')


    else:
        print("Please provide a valid directory or press enter.")
        raise NotADirectoryError(f"{"./res/" + user_input} is not a valid directory.")
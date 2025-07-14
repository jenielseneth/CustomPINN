import os
import torch
from matplotlib import pyplot as plt
from PINN import CustomPINN_Green2D, CustomPINN_Green2D_PoissonExplicit
from constants_utils import Hyperparameters, mesh_type
from dataset_utils import GreenPINNDataset, get_non_corners_mesh
from training_utils import InferenceUtils
from plot_utils import plot_multiple_points
from data_generation_utils import sample_points
from pde_utils import evaluate_greens_function_integral
from expr_generation_utils import expr_to_func, func_input_wrapper
from random_utils import find_line_with_keyword, retrieve_dict_from_json
from loss import fetch_quadrature_weights
from debugging_utils import check_poisson_2d_harmonic_func, plot_fundamentals, plot_greens_function_animation


if __name__ == "__main__":

    user_input = input("Enter the res folder we retrieve data from: ")
    main_dir = "./res/" + user_input + "/"
    if not os.path.exists(main_dir):
        raise IsADirectoryError(f'The directory {main_dir} does not exist.')

    user_input = input("Enter the model directory we get our model from: ")
    model_dir = main_dir + "models/" + user_input + "/"
    if not os.path.exists(model_dir):
        raise IsADirectoryError(f'The directory {model_dir} does not exist.')

    data_dir = main_dir + "data/"
    figure_dir = model_dir + "figures/"

    config_dict = retrieve_dict_from_json(model_dir + "config.json")
    config = Hyperparameters(**config_dict)
    test_data = GreenPINNDataset(data_file_path=data_dir, data_file_name="data_test.pt")
    domain = test_data.constants.domain
    training_utils = InferenceUtils(constants=test_data.constants, config=config)

    model = config.model_cls(**config.model_params)
    model.load_state_dict(torch.load(model_dir + "model_best_MSELoss().pth"))
    model.eval()

    universal_integration_mesh_size = test_data.constants.integration_mesh_size
    universal_integration_mesh_type: mesh_type = test_data.constants.integration_mesh_type
    universal_evaluation_mesh_size = test_data.constants.evaluation_mesh_size
    universal_evaluation_mesh_type: mesh_type = test_data.constants.evaluation_mesh_type





    harmonic_func_bool = True
    plot_fundamentals_bool = True
    greens_func_anim_bool = True

    # Check Poisson equation harmonic function for Greens Function of the form G(x, y) = log(r) + Psi(x, y)
    if harmonic_func_bool:
        eval_points = sample_points(domain=domain, 
                                    mesh_size=universal_evaluation_mesh_size, 
                                    mesh_type=universal_evaluation_mesh_type)
        harmonic_eval_points = get_non_corners_mesh(domain=domain, mesh=eval_points)
        
        integration_points = sample_points(domain=domain, 
                                            mesh_size=universal_integration_mesh_size, 
                                            mesh_type=universal_integration_mesh_type).expand(harmonic_eval_points.shape[0], -1, -1)
        
        quadrature_weights = fetch_quadrature_weights(domain=domain, integration_mesh_size=universal_integration_mesh_size, integration_mesh_type=universal_integration_mesh_type)

        check_poisson_2d_harmonic_func(model=model, 
                                       domain=domain,
                                       eval_points=harmonic_eval_points, 
                                       integration_points=integration_points, 
                                       quadrature_weights=quadrature_weights, 
                                       figure_dir=figure_dir)

    # Plot fundamentals
    if plot_fundamentals_bool:
        int_mesh_size = (50,50)
        int_mesh_type: mesh_type = "uniform"
        plot_fundamentals(integration_mesh_size=int_mesh_size, 
                          integration_mesh_type=int_mesh_type, 
                          domain=domain, 
                          data_constants=test_data.constants,
                          greens_function=model, 
                          config=config, 
                          figure_dir=figure_dir)
    
    # Greens Function animation
    if greens_func_anim_bool:
        
        uniform_mesh = sample_points(domain, mesh_size=(80,80), mesh_type="uniform")[None]

        def point_func(frames):
            x = domain[0]+frames/len(frames)
            y = domain[2]+frames/len(frames)
            return torch.vstack((x, y)).T

        plot_greens_function_animation(mesh=uniform_mesh, greens_function=model, 
                                    point_func=point_func, frames=40, title=str(config.model_cls),
                                    save_dir=figure_dir, save_name="GreensFuncAnim")
        
    
    # Psi Function animation
    if greens_func_anim_bool:
        uniform_mesh = sample_points(domain, mesh_size=(80,80), mesh_type="uniform")[None]

        def point_func(frames):
            x = domain[0]+frames/len(frames)
            y = domain[2]+frames/len(frames)
            return torch.vstack((x, y)).T

        plot_greens_function_animation(mesh=uniform_mesh, greens_function=lambda x, y: model.psi(x,y)[...,0], 
                                    point_func=point_func, frames=40, title="Ψ(x,y)",
                                    save_dir=figure_dir, save_name="PsiFunctionAnim")
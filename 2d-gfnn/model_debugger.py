import os
import torch
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
import math
from chebyshev_utils import cheb_2d_impl, cheb_2d_plot_debugger
from constants_utils import Hyperparameters, mesh_type
from dataset_utils import GreenPINNDataset, get_corners_idx, get_non_corners_mesh, get_interior_mesh
from training_utils import InferenceUtils
from plot_utils import plot_multiple_points
from data_generation_utils import sample_points
from pde_utils import evaluate_greens_function_integral, greens_function_laplacian_2d, u_laplacian_2d
from expr_generation_utils import expr_to_func, func_input_wrapper
from random_utils import find_line_with_keyword, retrieve_dict_from_json
from loss import fetch_quadrature_weights
from debugging_utils import check_poisson_2d_harmonic_func, plot_fundamentals, plot_greens_function_animation
import argparse

def harmonic_func_plotter(show: bool):
    '''
    Check Poisson equation harmonic function for Greens Function of the form G(x, y) = log(r) + Psi(x, y) 
    '''
    eval_points = sample_points(domain=domain, 
                                mesh_size=universal_evaluation_mesh_size, 
                                mesh_type=universal_evaluation_mesh_type)
    
    eval_points = sample_points(domain=domain, 
                                mesh_size=(80,80), 
                                mesh_type="uniform")
    harmonic_eval_points = get_non_corners_mesh(domain=domain, mesh=eval_points)
    
    integration_points = sample_points(domain=domain, 
                                        mesh_size=universal_integration_mesh_size, 
                                        mesh_type=universal_integration_mesh_type)
    
    quadrature_weights = fetch_quadrature_weights(domain=domain, integration_mesh_size=universal_integration_mesh_size, integration_mesh_type=universal_integration_mesh_type)

    check_poisson_2d_harmonic_func(model=model, 
                                    domain=domain,
                                    eval_points=harmonic_eval_points, 
                                    integration_points=integration_points, 
                                    quadrature_weights=quadrature_weights, 
                                    figure_dir=figure_dir, show=show)

def fundamentals_plotter(show: bool):
    '''
    Plot fundamentals
    '''

    int_mesh_size = (50,50)
    int_mesh_type: mesh_type = "uniform"
    plot_fundamentals(integration_mesh_size=int_mesh_size, 
                        integration_mesh_type=int_mesh_type, 
                        domain=domain, 
                        data_constants=test_data.constants,
                        greens_function=model, 
                        config=config, 
                        figure_dir=figure_dir,
                        show=show)
    
def sample_problems_plotter(show: bool):
    '''
    Sample problems plotter for debugging.
    '''
    # Sample 3 problems from the test data
    num=3
    size = math.prod(test_data.constants.evaluation_mesh_sizes)
    sample_gt = test_data[0:num*size]
    non_corner_idx, _ = get_corners_idx(domain=domain, mesh=sample_gt["crd"])
    eval_points = sample_gt["crd"][non_corner_idx].view(num, size-4, 2)
    u_gt = sample_gt["u_vals"][non_corner_idx].view(num, size-4)
    f_values = sample_gt["f_vals"][non_corner_idx].view(num, size-4, -1)
    f_meshes = sample_gt["f_mesh"][non_corner_idx].view(num, size-4, -1, 2)
    quadrature_weights = fetch_quadrature_weights(domain=domain,
                                                integration_mesh_size=test_data.constants.integration_mesh_sizes, 
                                                integration_mesh_type=test_data.constants.integration_mesh_type)
    approx_values = []
    for i in range(len(u_gt)):
        approx_values.append(evaluate_greens_function_integral(greens_function=model, evaluation_mesh=eval_points[i], integration_mesh_values=f_values[i],
                                      integration_meshes=f_meshes[i], quadrature_weights=quadrature_weights))
    mse = torch.nn.functional.mse_loss
    plot_multiple_points(points_list=[eval_points[0], eval_points[0], eval_points[0],  
                                      eval_points[1], eval_points[1], eval_points[1], 
                                      eval_points[2], eval_points[2], eval_points[2]], 
                    values_list=[u_gt[0], approx_values[0], mse(u_gt[0], approx_values[0], reduction="none"), 
                                 u_gt[1], approx_values[1], mse(u_gt[1], approx_values[1], reduction="none"),  
                                 u_gt[2], approx_values[2], mse(u_gt[2], approx_values[2], reduction="none"),], 
                    title_list=["u₁ᵍᵗ(x)",
                                "u₁ᵃᵖᵖʳᵒˣ(x)",
                                "L₁ = MSE(u₁ᵍᵗ(x)-u₁ᵃᵖᵖʳᵒˣ(x))",
                                "u₂ᵍᵗ(x)",
                                "u₂ᵃᵖᵖʳᵒˣ(x)",
                                "L₂ = MSE(u₂ᵍᵗ(x)-u₂ᵃᵖᵖʳᵒˣ(x))",
                                "u₃ᵍᵗ(x)",
                                "u₃ᵃᵖᵖʳᵒˣ(x)",
                                "L₃ = MSE(u₃ᵍᵗ(x)-u₃ᵃᵖᵖʳᵒˣ(x))",], 
                    main_title="Sample Problems from Test Data",
                    cmap_list=["viridis", "viridis", "plasma",
                               "viridis", "viridis", "plasma",
                               "viridis", "viridis", "plasma",],
                    axs_size=(3,3),
                    save_dir=figure_dir, save_name="SampleProblems",
                    log_info="Sample loss L = L₁ + L₂ + L₃ = " + f"{(mse(u_gt[0], approx_values[0]) + mse(u_gt[1], approx_values[1]) + mse(u_gt[2], approx_values[2])).item():.10f}",
                    figsize=(12, 10),
                    show=show)


def harmonic_loss_debugger():
    '''
    Calculate harmonic loss of u.
    '''
    # Sample 3 problems from the test data
    num=3
    size = math.prod(test_data.constants.evaluation_mesh_sizes)
    sample_gt = test_data[0:num*size]
    non_corner_idx, _ = get_corners_idx(domain=domain, mesh=sample_gt["crd"])
    eval_points = sample_gt["crd"][non_corner_idx].view(num, size-4, 2)
    u_gt = sample_gt["u_vals"][non_corner_idx].view(num, size-4)
    f_values = sample_gt["f_vals"][non_corner_idx].view(num, size-4, -1)
    f_meshes = sample_gt["f_mesh"][non_corner_idx].view(num, size-4, -1, 2)
    quadrature_weights = fetch_quadrature_weights(domain=domain,
                                                integration_mesh_size=test_data.constants.integration_mesh_sizes, 
                                                integration_mesh_type=test_data.constants.integration_mesh_type)

    loss = torch.tensor(0.)

    eval_meshes = []
    grad_vals = []
    f_itpl_point_values = []

    for i in range(num):

        assert test_data.constants.integration_mesh_type == "chebyshev", "Current implementation relies on integration mesh being chebyshev."
       
        eval_mesh = get_interior_mesh(domain=test_data.constants.domain, mesh=eval_points[i])
        eval_meshes.append(eval_mesh)

        f_eval_point_values = cheb_2d_impl(eval_points=eval_mesh,
                     chebyshev_size=test_data.constants.integration_mesh_sizes,
                     chebyshev_values=f_values[i] if f_values[i].dim() == 1 else f_values[i][0],
                     domain=test_data.constants.domain
                     )
        f_itpl_point_values.append(f_eval_point_values)
        
        grad_2_u = u_laplacian_2d(
            greens_function=model,
            x=eval_mesh, 
            s=f_meshes[i][0],
            s_values=f_values[i][0],
            quadrature_weights=quadrature_weights
            )
        grad_vals.append(grad_2_u)
        loss += torch.nn.functional.mse_loss(grad_2_u,f_eval_point_values)

    mse = torch.nn.functional.mse_loss

    plot_multiple_points(points_list=[eval_points[0], eval_meshes[0], eval_meshes[0], eval_meshes[0], 
                                      eval_points[1], eval_meshes[1], eval_meshes[1], eval_meshes[1], 
                                      eval_points[2], eval_meshes[2], eval_meshes[2], eval_meshes[2],],
                        values_list=[u_gt[0], -grad_vals[0], f_itpl_point_values[0], mse(-grad_vals[0], f_itpl_point_values[0], reduction="none"),
                                     u_gt[1], -grad_vals[1], f_itpl_point_values[1], mse(-grad_vals[1], f_itpl_point_values[1], reduction="none"),
                                     u_gt[2], -grad_vals[2], f_itpl_point_values[2], mse(-grad_vals[2], f_itpl_point_values[2], reduction="none"),],
                        cmap_list=["viridis", "plasma", "plasma", "inferno",
                                   "viridis", "plasma", "plasma", "inferno",
                                   "viridis", "plasma", "plasma", "inferno",],
                        title_list=["u₁ᵍᵗ(x)", "-∆u₁ᵃᵖᵖʳᵒˣ(x)", "f₁ᵍᵗ(x)", "MSE(∆u₁ᵃᵖᵖʳᵒˣ(x) + f₁ᵍᵗ(x))",
                                    "u₂ᵍᵗ(x)", "-∆u₂ᵃᵖᵖʳᵒˣ(x)", "f₂ᵍᵗ(x)", "MSE(∆u₂ᵃᵖᵖʳᵒˣ(x) + f₂ᵍᵗ(x))",
                                    "u₃ᵍᵗ(x)", "-∆u₃ᵃᵖᵖʳᵒˣ(x)", "f₃ᵍᵗ(x)", "MSE(∆u₃ᵃᵖᵖʳᵒˣ(x) + f₃ᵍᵗ(x))",],
                        axs_size=(3,4),
                        main_title="-∆u = f Debugger",
                        figsize=(18, 10),
                        log_info=f"L = L₁ + L₂ {f" + {config.harmonic_psi_loss_factor}L₃" if config.harmonic_psi_loss else ""} {f"+ {config.optimizer_params['weight_decay']}||w||²₂" if not config.optimizer_params["weight_decay"] == 0 else "" }",
                        )
    print(f"Loss of ||∆u-f|| is: {loss.item():.10f}")



def psi_harmonic_loss_debugger():
    '''
    Calculate harmonic loss of Ψ(x,s).
    '''
    size = math.prod(test_data.constants.evaluation_mesh_sizes)
    data = test_data[0:size]
    eval_points = data["crd"]

    # Define Psi function 
    def psi(x, s):
        # Expand x and s to match the expected input shape
        if x.dim() == 2:
            assert s.dim() == 2, "s must be a 2D tensor if x is 2D."
            x = x[:, None, :].expand(-1, s.shape[0], -1)  # b x f x 2 Tensor
            s = s[None, :, :].expand(x.shape[0], -1, -1)  # b x f x 2 Tensor

        return model.psi(x, s)[...,0]

    
    harmonic_term = greens_function_laplacian_2d(
        greens_function=psi,
        x=eval_points, 
        s=test_data.constants.integration_mesh
        )
    harmonic_loss = (harmonic_term**2).mean()
    print(f"Harmonic loss of ||∆Ψ(x,s)|| is: {harmonic_loss.item():.10f}")

    

def gf_diagonal_anim_plotter(show: bool):
    '''
    Greens Function diagonal animation
    '''
    uniform_mesh = sample_points(domain, mesh_size=(80,80), mesh_type="uniform")[None]

    def diagonal_point_func(frames):
        x = domain[0]+frames/len(frames)
        y = domain[2]+frames/len(frames)
        return torch.vstack((x, y)).T
    
    plot_greens_function_animation(mesh=uniform_mesh, 
                                   greens_function=model, 
                                   point_func=diagonal_point_func, 
                                   frames=40, 
                                   title=str(config.model_cls),
                                   save_dir=figure_dir, 
                                   save_name="GreensFuncAnim",
                                   show=show)
    

def gf_boundary_anim_plotter(show: bool):
    '''
    Greens Function boundary animation of x from evaluation mesh evaluated over an integration mesh. 
    '''
    uniform_mesh = sample_points(domain, mesh_size=(80, 80), mesh_type=universal_integration_mesh_type)[None]

    def boundary_point_func(frames_tensor):
        '''
        Parameters:
            frames_tensor (torch.Tensor): A tensor of shape (frames,) containing the frame indices.
        '''
        frames_fourth = len(frames_tensor) // 4
        x = torch.zeros_like(frames_tensor) + domain[0]
        y = torch.zeros_like(frames_tensor) + domain[2]
        # Move along the left edge of the domain 
        y[0:frames_fourth] = (domain[3]-domain[2]) * frames_tensor[0:frames_fourth] / frames_fourth + domain[2]
        # Move along the top edge of the domain
        x[frames_fourth:frames_fourth*2] = (domain[1]-domain[0]) * (frames_tensor[frames_fourth:frames_fourth*2]-frames_fourth) / frames_fourth + domain[0]
        y[frames_fourth:frames_fourth*2] = domain[3]
        # Move along the right edge of the domain
        x[frames_fourth*2:frames_fourth*3] = domain[1]
        y[frames_fourth*2:frames_fourth*3] = domain[3] - (domain[3]-domain[2]) * (frames_tensor[frames_fourth*2:frames_fourth*3]-frames_fourth*2) / frames_fourth
        # Move along the bottom edge of the domain
        x[frames_fourth*3:] = domain[1]-(domain[1]-domain[0]) * (frames_tensor[frames_fourth*3:]-frames_fourth*3) / len(frames_tensor[frames_fourth*3:])
        y[frames_fourth*3:] = domain[2]

        return torch.vstack((x, y)).T
    
    plot_greens_function_animation(mesh=uniform_mesh, 
                                   greens_function=lambda mesh, 
                                   point_func: model(point_func, mesh), 
                                   point_func=boundary_point_func, 
                                   frames=40, 
                                   title="G(x,s) over boundary x ∈ ∂Ω",
                                   save_dir=figure_dir, 
                                   save_name="GreensFuncBoundaryAnim",
                                   show=show)

def psi_func_anim_plotter(show: bool):
    '''
    Psi Function animation
    '''
    uniform_mesh = sample_points(domain, mesh_size=(80,80), mesh_type="uniform")[None]

    def diagonal_point_func(frames):
        x = domain[0]+frames/len(frames)
        y = domain[2]+frames/len(frames)
        return torch.vstack((x, y)).T

    plot_greens_function_animation(mesh=uniform_mesh, 
                                   greens_function=lambda mesh, 
                                   point_func: model.psi(mesh, point_func)[...,0], 
                                   point_func=diagonal_point_func, 
                                   frames=40, 
                                   title="Ψ(x,y)",
                                   save_dir=figure_dir, 
                                   save_name="PsiFunctionAnim",
                                   show=show)
    
def int_mesh_gfxw_anim_plotter(show: bool):
    '''
    Green Function times quadrature weights over integration mesh evaluated at each evaluation mesh point animation
    '''
    mesh = sample_points(domain, mesh_size=universal_integration_mesh_size, mesh_type=universal_integration_mesh_type)[None]

    def delta_function_singularity(frames):
        points = sample_points(domain, mesh_size=universal_evaluation_mesh_size, mesh_type=universal_evaluation_mesh_type)
        return points

    plot_greens_function_animation(mesh=mesh, 
                                    greens_function =lambda mesh, point_func: model(point_func,mesh)*fetch_quadrature_weights(domain=domain, 
                                                                                                    integration_mesh_size=universal_integration_mesh_size, 
                                                                                                    integration_mesh_type=universal_integration_mesh_type), 
                                    point_func=delta_function_singularity, 
                                    frames=math.prod(test_data.constants.evaluation_mesh_sizes), 
                                    title="G(x,s)w(s), s ∈ Ω",
                                    save_dir=figure_dir, save_name="GFxWIntegrationMeshAnim",
                                    show=show)

def int_mesh_gf_anim_plotter(show: bool):
    '''
    Green Function over integration mesh evaluated at each evaluation mesh point animation
    '''
    mesh = sample_points(domain, mesh_size=test_data.constants.integration_mesh_sizes, mesh_type=test_data.constants.integration_mesh_type)[None]

    def delta_function_singularity(frames):
        points = sample_points(domain, mesh_size=test_data.constants.evaluation_mesh_sizes, mesh_type=test_data.constants.evaluation_mesh_type)
        return points

    plot_greens_function_animation(mesh=mesh, 
                                    greens_function =lambda mesh, point_func: model(point_func,mesh), 
                                    point_func=delta_function_singularity, 
                                    frames=math.prod(test_data.constants.evaluation_mesh_sizes), 
                                    title="G(x,s), s ∈ Ω",
                                    save_dir=figure_dir, save_name="GFIntegrationMeshAnim",
                                    show=show)
    

def data_visualiser():
    train_data = GreenPINNDataset(data_file_path=main_dir, data_file_name="data_train.pt")
    train_sample_1 = train_data[slice(*train_data.u_data_addresses[0])]
    train_sample_2 = train_data[slice(*train_data.u_data_addresses[2])]
    train_sample_3 = train_data[slice(*train_data.u_data_addresses[3])]
    plot_multiple_points(points_list=[train_sample_1["crd"], train_sample_1["f_mesh"],
                                      train_sample_2["crd"], train_sample_2["f_mesh"],
                                      train_sample_3["crd"], train_sample_3["f_mesh"]],
                        values_list=[train_sample_1["u_vals"], train_sample_1["f_vals"],
                                     train_sample_2["u_vals"], train_sample_2["f_vals"],
                                     train_sample_3["u_vals"], train_sample_3["f_vals"],],
                        cmap_list=["viridis", "plasma",
                                   "viridis", "plasma",
                                   "viridis", "plasma",],
                        title_list=["Train Sample 1 u(x)", "Train Sample 1 f(x)",
                                    "Train Sample 2 u(x)", "Train Sample 2 f(x)",
                                    "Train Sample 3 u(x)",  "Train Sample 3 f(x)"],
                    axs_size=(3,2),
                    main_title="Train Data Visualisation",
                    figsize=(18, 10),
                    )

if __name__ == "__main__":
    anims = {
    'diag': gf_diagonal_anim_plotter,
    'bound': gf_boundary_anim_plotter,
    'psi': psi_func_anim_plotter,
    'int': int_mesh_gf_anim_plotter,
    'int_gfxw': int_mesh_gfxw_anim_plotter,
    }

    plots = {
    'harm_plot': harmonic_func_plotter,
    'fund': fundamentals_plotter,
    'sample': sample_problems_plotter,
    }
    
    debuggers = {
        "u_harm": harmonic_loss_debugger,
        "psi_harm": psi_harmonic_loss_debugger
    }

    data = {
        "data_vis": data_visualiser
    }

    tasks = {
        **anims,
        **plots,
        **debuggers,
        **data
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--rd', type=str, required=True, help='Which res folder to use.')
    parser.add_argument('--md', nargs='+', type=str, required=True, help='Which model folder to use.')
    # Which tests to run
    parser.add_argument('--all', action='store_true', help='Run all tests.') 
    parser.add_argument('--anims', action='store_true', help='Run all animation tests.') 
    parser.add_argument('--plots', action='store_true', help='Run all plot tests.') 
    parser.add_argument('--debuggers', action='store_true', help='Run all debuggers tests.') 
    parser.add_argument('--data', action='store_true', help='Run all data checker tests.') 
    parser.add_argument('--run', nargs='+', choices=tasks.keys(), help="Tasks to run. Choose from: " + ", ".join(tasks.keys()))
    parser.add_argument('--show', action='store_true', help='Show plots.') 
    args = parser.parse_args()

    main_dir = "./res/" + args.rd + "/"
    if not os.path.exists(main_dir):
        raise IsADirectoryError(f'The directory {main_dir} does not exist.')
    
    for md in args.md:
        print(f"Running debugger for model directory: {md}")
        model_dir = main_dir + "models/" + md + "/"
        if not os.path.exists(model_dir):
            raise IsADirectoryError(f'The directory {model_dir} does not exist.')

        data_dir = main_dir + "data/"
        figure_dir = model_dir + "figures/"
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)

        config_dict = retrieve_dict_from_json(model_dir + "config.json")
        config = Hyperparameters(**config_dict)
        test_data = GreenPINNDataset(data_file_path=data_dir, data_file_name="data_test.pt")
        domain = test_data.constants.domain

        model = config.model_cls(**config.model_params)
        model.load_state_dict(torch.load(model_dir + "model_best_prediction_MSELoss().pth"))
        model.eval()

        universal_integration_mesh_size = test_data.constants.integration_mesh_sizes
        universal_integration_mesh_type: mesh_type = test_data.constants.integration_mesh_type
        universal_evaluation_mesh_size = test_data.constants.evaluation_mesh_sizes
        universal_evaluation_mesh_type: mesh_type = test_data.constants.evaluation_mesh_type
        
        if args.anims:
            for func in tqdm(anims.values(), "Running all animation tests..."):
                func(args.show)
        
        if args.plots:
            for func in tqdm(plots.values(), "Running all plot tests..."):
                func(args.show)

        if args.data:
            for func in tqdm(data.values(), "Running all data tests..."):
                func()
        
        if args.all:
            for key, func in tqdm(tasks.items(), "Running all tests..."):
                if not key == "harm":
                    func(args.show)
                else: 
                    func()

        if args.run is not None:
            for task_key in args.run:
                if not (task_key == "u_harm" or task_key == "psi_harm"):
                    tasks[task_key](args.show)  # Call the corresponding function
                else:
                    tasks[task_key]()  # Call the corresponding function


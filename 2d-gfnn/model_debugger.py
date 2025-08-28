import os
from ngsolve import *
import torch
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
import math
from chebyshev_utils import cheb_2d_impl, cheb_2d_plot_debugger
from constants_utils import Hyperparameters, mesh_type
from dataset_utils import GreenPINNDataset, get_corners_idx, get_non_corners_mesh, get_interior_mesh
from plot_utils import plot_multiple_points
from data_generation_utils import sample_points
from pde_utils import InferenceUtils, evaluate_greens_function_integral, greens_function_laplacian_2d, u_laplacian_2d, StandardizedInferenceUtils
from expr_generation_utils import expr_to_func, func_input_wrapper
from random_utils import find_line_with_keyword, retrieve_dict_from_json
from loss import fetch_quadrature_weights
from debugging_utils import check_poisson_2d_harmonic_func, plot_fundamentals, plot_greens_function_animation
import argparse
import logging

def harmonic_func_plotter():
    '''
    Check Poisson equation harmonic function for Greens Function of the form G(x, y) = log(r) + Psi(x, y) 
    '''
    eval_points = sample_points(domain=domain, 
                                mesh_size=global_evaluation_mesh_size, 
                                mesh_type=global_evaluation_mesh_type)
    
    eval_points = sample_points(domain=domain, 
                                mesh_size=(80,80), 
                                mesh_type="uniform")
    harmonic_eval_points = get_non_corners_mesh(domain=domain, mesh=eval_points)
    
    integration_points = sample_points(domain=domain, 
                                        mesh_size=global_integration_mesh_size, 
                                        mesh_type=global_integration_mesh_type)
    
    quadrature_weights = fetch_quadrature_weights(domain=domain, integration_mesh_size=global_integration_mesh_size, integration_mesh_type=global_integration_mesh_type)

    check_poisson_2d_harmonic_func(model=model, 
                                    domain=domain,
                                    eval_points=harmonic_eval_points, 
                                    integration_points=integration_points, 
                                    quadrature_weights=quadrature_weights, 
                                    figure_dir=figure_dir, show=show)

def fundamentals_plotter():
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
    
def sample_problems_plotter():
    '''
    Sample problems plotter for debugging.
    '''
    # Sample 3 problems from the test data
    num=3
    if len(test_data.f_meshes) < num:
        random_f_mesh_idx = random.choices(range(0, len(test_data.f_meshes)), k=num)
    elif len(test_data.f_meshes) == num:
        random_f_mesh_idx = range(0, num)
    else:
        random_f_mesh_idx = random.sample(range(0, len(test_data.f_meshes)), num)
    random_f_values_idx = [random.sample(range(0, len(test_data.f_values[idx])), 1)[0] for idx in random_f_mesh_idx]
    approx_values = []
    u_gt = []
    eval_points = []

    inference_utils = StandardizedInferenceUtils(constants=test_data.constants, config=config)
    # size = math.prod(universal_evaluation_mesh_size)
    for f_m_i, f_v_i in zip(random_f_mesh_idx, random_f_values_idx):
        u_data_address = test_data.u_data_addresses[f_m_i][f_v_i]
        sample_gt = test_data[slice(*u_data_address)] # num x size x 2 List of Dataset objects
        non_corner_idx, _ = get_corners_idx(domain=domain, mesh=sample_gt.crd)
        crd =sample_gt.crd[non_corner_idx] 
        gt = sample_gt.u_vals[non_corner_idx]
        f_mesh = global_f_meshes[f_m_i]
        f_values = sample_gt.f_vals[non_corner_idx]
        
        # quadrature_weights = fetch_quadrature_weights(domain=domain,
        #                                             integration_mesh_size=test_data.constants.integration_mesh_sizes[f_m_i], 
        #                                             integration_mesh_type=test_data.constants.integration_mesh_type)

        quadrature_weights = inference_utils.quadrature_weights[f_m_i]

        approx_values.append(evaluate_greens_function_integral(greens_function=model, 
                                                               evaluation_mesh=crd, 
                                                               integration_mesh_values=f_values,
                                                               integration_meshes=f_mesh, 
                                                               quadrature_weights=quadrature_weights))
        u_gt.append(gt)
        eval_points.append(crd)
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

def quadrature_rule_debugger():
    '''
    Checks quadrature rule approximation for varying resolutions.
    '''

    # ---- 1. Calculate for a randomly generated problem the f_values for different f_mesh resolutions, 
    #   along with the ground-truth u(x) values for a set of evaluation points. ----
    f_mesh_points = []
    quad_weights = []

    # Assure f_mesh_points are of original size and have no padding
    for i, mesh in enumerate(test_data.f_meshes):
        f_mesh_points.append(mesh[:math.prod(test_data.constants.integration_mesh_sizes[i])])
    

    # Assure quad_weights are of original size and have no padding
    for i, weights in enumerate(test_inference_utils.quadrature_weights):
        quad_weights.append(weights[:math.prod(test_data.constants.integration_mesh_sizes[i])])


    f_mesh_values = []

    mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))

    # H1-conforming finite element space
    fes = H1(mesh, order=3, dirichlet=[1,2,3,4])

    # define trial- and test-functions
    u = fes.TrialFunction()
    v = fes.TestFunction()

    # the bilinear-form 
    a = BilinearForm(fes, symmetric=True)
    a += grad(u)*grad(v)*dx
    a.Assemble()

    f = LinearForm(fes)
    coeff_a = random.uniform(1, 10)
    sigma_x = random.uniform(0.01, 0.5)
    sigma_y = random.uniform(0.01, 0.5)
    mean_x = random.uniform(domain[0], domain[1])
    mean_y = random.uniform(domain[2], domain[3])

    cf = CoefficientFunction(coeff_a*exp(-((x-mean_x)**2 / (2*sigma_x**2) + (y-mean_y)**2 / (2*sigma_y**2))))

    for points in f_mesh_points:
        integration_values = []
        for point in points:
            integration_values.append(cf(mesh(*point)))
        integration_values = torch.tensor(integration_values)
        f_mesh_values.append(integration_values)

    f += cf * v * dx
    f.Assemble()

    gfu = GridFunction(fes)
    gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec
    
    eval_points = get_interior_mesh(domain=test_data.constants.domain, 
                                    mesh=sample_points(test_data.constants.domain,
                                mesh_size=test_data.constants.evaluation_mesh_sizes[-1],
                                mesh_type=test_data.constants.evaluation_mesh_type))
    gt_values = []
    for point in eval_points:
        mesh_point = mesh(*point)
        gt_values.append(gfu(mesh_point))

    gt_values = torch.tensor(gt_values)

    #--- 2. Calculate using the given model the approximation given the different resolutions for a given quadrature rule. ----
    u_approx_values = []
    for i, (points, values) in enumerate(zip(f_mesh_points, f_mesh_values)):
        u_approx = evaluate_greens_function_integral(greens_function=model,
                                      evaluation_mesh=eval_points,
                                      integration_meshes=points,
                                      integration_mesh_values=values,
                                      quadrature_weights=quad_weights[i])
        u_approx_values.append(u_approx)
    u_approx_values = torch.stack(u_approx_values)
    logger.info("Quadrature rule debugger - approximating ground truth solution for varying integration mesh resolutions...")
    for i, mesh in enumerate(f_mesh_points):
        logger.info(f"For f_mesh of size ({mesh.shape}), MSELoss of u_gt compared against u_approx = {torch.nn.functional.mse_loss(u_approx_values[i], gt_values)}")
        


def harmonic_loss_debugger():
    '''
    Calculate harmonic loss of u.
    '''
    assert test_data.constants.integration_mesh_type == "chebyshev", "Current implementation relies on integration mesh being chebyshev."
    
    # Sample num problems from the test data
    num=3
    if len(test_data.f_meshes) < num:
        random_f_mesh_idx = random.choices(range(0, len(test_data.f_meshes)), k=num)
    elif len(test_data.f_meshes) == num:
        random_f_mesh_idx = range(0, num)
    else:
        random_f_mesh_idx = random.sample(range(0, len(test_data.f_meshes)), num)
    # Choose 3 random f_meshes and respective source terms. 
    random_f_values_idx = [random.sample(range(0, len(test_data.f_values[idx])), 1)[0] for idx in random_f_mesh_idx]
    u_gt = []

    loss = torch.tensor(0.)
    # Store ground truth evaluation meshes for plotting
    gt_eval_meshes = []
    # Store interior evaluation meshes for plotting
    interior_eval_meshes = []
    grad_vals = []
    f_itpl_point_values = []

    for f_m_i, f_v_i in zip(random_f_mesh_idx, random_f_values_idx):
        u_data_address = test_data.u_data_addresses[f_m_i][f_v_i]
        sample_gt = test_data[slice(*u_data_address)] # num x size x 2 List of Dataset objects
        non_corner_idx, _ = get_corners_idx(domain=domain, mesh=sample_gt.crd)
        crd =sample_gt.crd[non_corner_idx] 
        gt = sample_gt.u_vals[non_corner_idx]
        f_mesh = global_f_meshes[f_m_i]
        f_values = sample_gt.f_vals[non_corner_idx]
        quadrature_weights = test_inference_utils.quadrature_weights[f_m_i]
        gt_eval_meshes.append(crd)
        u_gt.append(gt)

        interior_eval_mesh = get_interior_mesh(domain=test_data.constants.domain, mesh=crd)
        interior_eval_meshes.append(interior_eval_mesh)
        
        # Interpolate f values onto the evaluation mesh
        f_eval_point_values = cheb_2d_impl(eval_points=interior_eval_mesh,
                     chebyshev_size=test_data.constants.integration_mesh_sizes[f_m_i],
                     chebyshev_values=f_values[0][:math.prod(test_data.constants.integration_mesh_sizes[f_m_i])],
                     domain=test_data.constants.domain
                     )
        
        grad_2_u = u_laplacian_2d(
            greens_function=model,
            x=interior_eval_mesh, 
            s=f_mesh,
            s_values=f_values[0],
            quadrature_weights=quadrature_weights
            )
        grad_vals.append(grad_2_u)
        loss += torch.nn.functional.mse_loss(grad_2_u,f_eval_point_values)

        f_itpl_point_values.append(f_eval_point_values)

    mse = torch.nn.functional.mse_loss

    plot_multiple_points(points_list=[gt_eval_meshes[0], interior_eval_meshes[0], interior_eval_meshes[0], interior_eval_meshes[0], 
                                      gt_eval_meshes[1], interior_eval_meshes[1], interior_eval_meshes[1], interior_eval_meshes[1], 
                                      gt_eval_meshes[2], interior_eval_meshes[2], interior_eval_meshes[2], interior_eval_meshes[2],],
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
                        save_dir=figure_dir,
                        save_name="HarmonicLossDebugger",
                        log_info=f"L = L₁ + L₂ {f" + {config.harmonic_psi_loss_factor}L₃" if config.harmonic_psi_loss else ""} {f"+ {config.optimizer_params['weight_decay']}||w||²₂" if not config.optimizer_params["weight_decay"] == 0 else "" }",
                        )
    print(f"Loss of ||∆u-f|| is: {loss.item():.10f}")



def psi_harmonic_loss_debugger():
    '''
    Calculate harmonic loss of Ψ(x,s).
    '''
    # Retrieve final source term for last f_mesh.
    u_address = test_data.u_data_addresses[-1][-1]
    data = test_data[slice(*u_address)]
    eval_points = data.crd
    f_mesh = global_f_meshes[-1]

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
        s=f_mesh
        )
    harmonic_loss = (harmonic_term**2).mean()
    print(f"Harmonic loss of ||∆Ψ(x,s)|| is: {harmonic_loss.item():.10f}")

    

def gf_diagonal_anim_plotter():
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
    

def gf_boundary_anim_plotter():
    '''
    Greens Function boundary animation of x from evaluation mesh evaluated over an integration mesh. 
    '''
    uniform_mesh = sample_points(domain, mesh_size=(80, 80), mesh_type=global_integration_mesh_type)[None]

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

def psi_func_anim_plotter():
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
    
def int_mesh_gfxw_anim_plotter():
    '''
    Green Function times quadrature weights over integration mesh evaluated at each evaluation mesh point animation
    '''
    mesh = sample_points(domain, mesh_size=global_integration_mesh_size, mesh_type=global_integration_mesh_type)[None]

    def delta_function_singularity(frames):
        points = sample_points(domain, mesh_size=global_evaluation_mesh_size, mesh_type=global_evaluation_mesh_type)
        return points

    plot_greens_function_animation(mesh=mesh, 
                                    greens_function =lambda mesh, point_func: model(point_func,mesh)*fetch_quadrature_weights(domain=domain, 
                                                                                                    integration_mesh_size=global_integration_mesh_size, 
                                                                                                    integration_mesh_type=global_integration_mesh_type), 
                                    point_func=delta_function_singularity, 
                                    frames=math.prod(global_evaluation_mesh_size), 
                                    title="G(x,s)w(s), s ∈ Ω",
                                    save_dir=figure_dir, save_name="GFxWIntegrationMeshAnim",
                                    show=show)

def int_mesh_gf_anim_plotter():
    '''
    Green Function over integration mesh evaluated at each evaluation mesh point animation
    '''
    mesh = sample_points(domain, mesh_size=global_integration_mesh_size, mesh_type=global_integration_mesh_type)[None]

    def delta_function_singularity(frames):
        points = sample_points(domain, mesh_size=global_evaluation_mesh_size, mesh_type=global_evaluation_mesh_type)
        return points

    plot_greens_function_animation(mesh=mesh, 
                                    greens_function =lambda mesh, point_func: model(point_func,mesh), 
                                    point_func=delta_function_singularity, 
                                    frames=math.prod(global_evaluation_mesh_size), 
                                    title="G(x,s), s ∈ Ω",
                                    save_dir=figure_dir, save_name="GFIntegrationMeshAnim",
                                    show=show)


if __name__ == "__main__":

    # Set logging format.
    logging.basicConfig(level=logging.INFO,
    format="%(filename)s:%(lineno)d - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Set parser flags.
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
        "psi_harm": psi_harmonic_loss_debugger,
        "quad": quadrature_rule_debugger

    }

    tasks = {
        **anims,
        **plots,
        **debuggers,
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

        if config.test_dir is not None:
            data_dir = "./res/" + config.test_dir + "/data/"
        test_data = GreenPINNDataset(data_file_path=data_dir, 
                                     config=config,
                                     data_file_name="data_test.pt")
        test_inference_utils = StandardizedInferenceUtils(constants=test_data.constants, config=config) if config.multi_mesh_training_variant == "standardize" else InferenceUtils(constants=test_data.constants, config=config)
        global_f_meshes = test_data.f_meshes
        domain = test_data.constants.domain

        show = args.show

        model = config.model_cls(**config.model_params)
        model.load_state_dict(torch.load(model_dir + "model_best_prediction_MSELoss().pth"))
        model.eval()

        global_integration_mesh_size = test_data.constants.integration_mesh_sizes[-1]
        global_integration_mesh_type: mesh_type = test_data.constants.integration_mesh_type
        global_evaluation_mesh_size = test_data.constants.evaluation_mesh_sizes[-1]
        global_evaluation_mesh_type: mesh_type = test_data.constants.evaluation_mesh_type
        
        if args.anims:
            for func in tqdm(anims.values(), "Running all animation tests..."):
                func(args.show)
        
        if args.plots:
            for func in tqdm(plots.values(), "Running all plot tests..."):
                func(args.show)

        if args.all:
            for key, func in tqdm(tasks.items(), "Running all tests..."):
                func()

        if args.run is not None:
            for task_key in args.run:
                tasks[task_key]()  # Call the corresponding function


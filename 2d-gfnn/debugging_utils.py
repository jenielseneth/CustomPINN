# solve the Poisson equation -Delta u = f
# with Dirichlet boundary condition u = 0
from matplotlib import animation, pyplot as plt
from ngsolve import *
from pde_utils import evaluate_greens_function_integral
from plot_utils import plot_points, plot_multiple_points
from data_generation_utils import sample_points
from loss import fetch_quadrature_weights
from constants_utils import mesh_type, Hyperparameters
from dataset_utils import GreensConstantsDataclass, get_interior_boundary_idx
import random, torch
from netgen.csg import *

ngsglobals.msg_level = 1


def check_poisson_2d_harmonic_func(model, eval_points, integration_points, quadrature_weights, domain, figure_dir, show: bool = True):
    '''
    Parameters:
        model (Callable): The model to evaluate.
        eval_points (b x 2 torch.Tensor): The points to evaluate the model at.
        integration_points (f x 2 torch.Tensor): The points to integrate the model over.
        quadrature_weights (f torch.Tensor): The quadrature weights for the integration points.
        domain (tuple): The domain of the problem.
        figure_dir (str): The directory to save the figures to.
    
    '''
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

    integration_values = []
    for point in integration_points:
        integration_values.append(cf(mesh(*point)))
    integration_values = torch.tensor(integration_values)

    f += cf * v * dx
    f.Assemble()

    gfu = GridFunction(fes)
    gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec
    
    
    gt_values = []
    for point in eval_points:
        mesh_point = mesh(*point)
        gt_values.append(gfu(mesh_point))

    gt_values = torch.tensor(gt_values)

    approx_values = evaluate_greens_function_integral(greens_function=model, evaluation_mesh=eval_points, integration_mesh_values=integration_values,
                                      integration_meshes=integration_points, quadrature_weights=quadrature_weights)
    approx_log_values = evaluate_greens_function_integral(greens_function=lambda x, s: torch.log(torch.sqrt(((x-s)**2).sum(-1))), evaluation_mesh=eval_points, integration_mesh_values=integration_values,
                                      integration_meshes=integration_points, quadrature_weights=quadrature_weights)
    approx_psi_values = evaluate_greens_function_integral(greens_function=lambda x, s: model.psi(x, s)[..., 0], evaluation_mesh=eval_points, integration_mesh_values=integration_values,
                                      integration_meshes=integration_points, quadrature_weights=quadrature_weights)
    
    int_idx, bnd_idx = get_interior_boundary_idx(domain=domain, mesh=eval_points)
        
    integration_values_times_weights = integration_values * quadrature_weights

    log_info = "|G(x,s)| on boundary: " + str(torch.sum(torch.abs(approx_values[bnd_idx])).item()) 
    plot_multiple_points(points_list=[eval_points, eval_points, eval_points, eval_points, integration_points, integration_points], 
                         values_list=[approx_psi_values, approx_log_values, approx_values, gt_values, integration_values, integration_values_times_weights], 
                         title_list=["h(x) ≈ ∫Ψ(x,s)f(s)ds",  "∫log(|x-s|)f(s)ds", "uᵃᵖᵖʳᵒˣ(x) ≈ ∫G(x,s)f(s)ds", "uᵍᵗ(x)", "f(s) over Ω", "f(s) * w(s) over Ω"], 
                         cmap_list=["viridis", "plasma", "viridis", "viridis", "viridis", "plasma"],
                         main_title="Harmonic function h(x) Analysis",
                         axs_size=(3,2),
                         log_info=log_info, 
                         save_dir=figure_dir, save_name="Harmonic_Function",
                         show=show)


def plot_fundamentals(integration_mesh_size: tuple, 
                      integration_mesh_type: mesh_type, 
                      domain: tuple, 
                      data_constants: GreensConstantsDataclass, 
                      greens_function, 
                      config: Hyperparameters, 
                      figure_dir: str,
                      show: bool = True):

    mesh = sample_points(domain, mesh_size=integration_mesh_size, mesh_type=integration_mesh_type)[None, :, :]

    if config.l_weights:
        weights_uniform = greens_function.quadrature_weights(mesh)**2
    else: 
        weights_uniform = fetch_quadrature_weights(data_constants.domain, 
                                                integration_mesh_size=integration_mesh_size, 
                                                integration_mesh_type=data_constants.integration_mesh_type)

    domain_center = torch.tensor(((domain[1]-domain[0])/2, (domain[3]-domain[2])/2))
    domain_center_mesh = torch.zeros_like(mesh) + domain_center
    domain_edge = torch.tensor((domain[1], domain[3]))
    domain_edge_mesh = torch.zeros_like(mesh) + domain_edge

    psi_uniform = greens_function.psi(mesh, domain_center_mesh)[0, :, 0]
    phi_uniform = greens_function.phi(mesh, domain_center_mesh)[0, :, 0]

    log_term = torch.log(torch.sqrt(((domain_center_mesh - mesh)**2).sum(-1)))[0]
    model_term_center = greens_function(mesh, domain_center_mesh)[0] # Remove batch dimension for plotting
    model_term_edge = greens_function(mesh, domain_edge_mesh)[0] # Remove batch dimension for plotting

    mesh = mesh[0]  # Remove batch dimension for plotting

    plot_multiple_points(points_list=[mesh, mesh, mesh, mesh, mesh, mesh], 
                        values_list=[weights_uniform, psi_uniform, phi_uniform, log_term, model_term_center, model_term_edge], 
                        title_list=["Quadrature weights", "Ψ(x,s), s = domain center",
                                    "Φ(x,s), s = domain center", "log(|x-s|), s = domain center", "G(x,s), s = domain center", "G(x,s), s = top right corner"], 
                        cmap_list=["plasma", "viridis","viridis", "plasma", "viridis","viridis"],
                        main_title="Model Decomposition Plots: G(x,s) = Φ(x,s)log(|x-s|) + Ψ(x,s)",
                        axs_size=(3,2),
                        save_dir=figure_dir, save_name="ModelDecomposition",
                        show=show)
    

    
def plot_greens_function_animation(mesh, 
                                   greens_function, 
                                   point_func, 
                                   frames=40, 
                                   cmap='viridis', 
                                   title="", 
                                   save_dir = None, 
                                   save_name = None, 
                                   vmax=0.4,
                                   show: bool = True):
    '''
    Plot an animation of the approximated Green's Function.

    Parameters:
        mesh (1 x b x 2 size torch.Tensor) : The mesh points to evaluate the Green's Function on.
        greens_function: (mesh, point_func) (Callable)
        point_func (Callable): Define which points to evaluate per frame.
    '''
    fig, ax = plt.subplots()
    ax.set_title(title)
    points = point_func(torch.linspace(1, frames, frames))
    sc = ax.scatter(mesh[0][:,0].detach().numpy(), mesh[0][:,1].detach().numpy(), 
                    c=greens_function(mesh, torch.zeros_like(mesh) + points[0]).detach().numpy(), 
                    cmap=cmap)
    point_line = ax.scatter(*points[0], c="red")
    cbar = plt.colorbar(sc, ax=ax)  # create colorbar
    def update(frame):
        # for each frame, update the data stored on each artist.
        sc.set_array(greens_function(mesh, torch.zeros_like(mesh) + points[frame])[0].detach().numpy())
        point_line.set_offsets(points[frame])

    ani = animation.FuncAnimation(fig=fig, func=update, frames=frames, interval=30)
    if show:
        plt.show()
    if save_dir is not None and save_name is not None:
        save_name = save_name + ".gif"
        ani.save(filename=save_dir + save_name, writer='pillow', fps=10, dpi=80)

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


def check_poisson_2d_harmonic_func(model, eval_points, integration_points, quadrature_weights, domain, figure_dir):
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
    for point in integration_points[0]:
        integration_values.append(cf(mesh(*point)))
    integration_values = torch.tensor(integration_values).expand(eval_points.shape[0], -1)

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
                                      integration_mesh=integration_points, quadrature_weights=quadrature_weights)
    approx_log_values = evaluate_greens_function_integral(greens_function=lambda x, y: torch.log(torch.sqrt(((x-y)**2).sum(-1))), evaluation_mesh=eval_points, integration_mesh_values=integration_values,
                                      integration_mesh=integration_points, quadrature_weights=quadrature_weights)
    approx_psi_values = evaluate_greens_function_integral(greens_function=lambda x, y: model.psi(x, y)[..., 0], evaluation_mesh=eval_points, integration_mesh_values=integration_values,
                                      integration_mesh=integration_points, quadrature_weights=quadrature_weights)
    
    int_idx, bnd_idx = get_interior_boundary_idx(domain=domain, mesh=eval_points)

    log_info = "|log(|x-y|)+Ψ(x,y)| on boundary: " + str(torch.sum(torch.abs(approx_log_values[bnd_idx]+approx_psi_values[bnd_idx])).item()) 
    print(integration_values.shape, gt_values.shape, eval_points.shape)
    plot_multiple_points(points_list=[eval_points, eval_points, eval_points, eval_points, integration_points[0]], 
                         values_list=[approx_psi_values, approx_log_values, approx_values, gt_values, integration_values[0]], 
                         title_list=["h(x) ≈ ∫Ψ(x,y)f(y)dy",  "∫log(|x-y|)f(y)dy", "uᵃᵖᵖʳᵒˣ(x) ≈ ∫G(x,y)f(y)dy", "uᵍᵗ(x)", "f(x) over Ω",], 
                         cmap_list=["viridis", "plasma", "viridis", "viridis", "viridis"],
                         main_title="Harmonic function h(x) Analysis",
                         axs_size=(3,2),
                         log_info=log_info, 
                         save_dir=figure_dir, save_name="Harmonic_Function")

def check_sample_solutions(model, eval_points, integration_points, quadrature_weights, domain, figure_dir):
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
    for point in integration_points[0]:
        integration_values.append(cf(mesh(*point)))
    integration_values = torch.tensor(integration_values).expand(eval_points.shape[0], -1)

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
                                      integration_mesh=integration_points, quadrature_weights=quadrature_weights)
    approx_log_values = evaluate_greens_function_integral(greens_function=lambda x, y: torch.log(torch.sqrt(((x-y)**2).sum(-1))), evaluation_mesh=eval_points, integration_mesh_values=integration_values,
                                      integration_mesh=integration_points, quadrature_weights=quadrature_weights)
    approx_psi_values = evaluate_greens_function_integral(greens_function=lambda x, y: model.psi(x, y)[..., 0], evaluation_mesh=eval_points, integration_mesh_values=integration_values,
                                      integration_mesh=integration_points, quadrature_weights=quadrature_weights)
    
    int_idx, bnd_idx = get_interior_boundary_idx(domain=domain, mesh=eval_points)

    log_info = "|log(|x-y|)+Ψ(x,y)| on boundary: " + str(torch.sum(torch.abs(approx_log_values[bnd_idx]+approx_psi_values[bnd_idx])).item()) 

    plot_multiple_points(points_list=[eval_points, eval_points, eval_points, eval_points], 
                         values_list=[gt_values, approx_psi_values, approx_log_values, approx_values], 
                         title_list=["f(x) over Ω", "h(x) ≈ ∫Ψ(x,y)f(y)dy",  "∫log(|x-y|)f(y)dy", "u(x) ≈ ∫G(x,y)f(y)dy"], 
                         cmap_list=["viridis", "viridis", "plasma", "viridis"],
                         main_title="Harmonic function h(x) Analysis",
                         axs_size=(2,2),
                         log_info=log_info, 
                         save_dir=figure_dir, save_name="Harmonic_Function")


def plot_fundamentals(integration_mesh_size: tuple, integration_mesh_type: mesh_type, domain: tuple, data_constants: GreensConstantsDataclass, greens_function, config: Hyperparameters, figure_dir: str):

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
    model_term_center = greens_function(domain_center_mesh, mesh)[0] # Remove batch dimension for plotting
    model_term_edge = greens_function(domain_edge_mesh, mesh)[0] # Remove batch dimension for plotting

    mesh = mesh[0]  # Remove batch dimension for plotting

    plot_multiple_points(points_list=[mesh, mesh, mesh, mesh, mesh, mesh], 
                        values_list=[weights_uniform, psi_uniform, phi_uniform, log_term, model_term_center, model_term_edge], 
                        title_list=["Quadrature weights", "Ψ(x,y), x = domain center",
                                    "Φ(x,y), x = domain center", "log(|x-y|), x = domain center", "G(x,y), x = domain center", "G(x,y) x = top right corner"], 
                        cmap_list=["plasma", "viridis","viridis", "plasma", "viridis","viridis"],
                        main_title="Model Decomposition Plots: G(x,y) = Φ(x,y)log(|x-y|) + Ψ(x,y)",
                        axs_size=(3,2),
                        save_dir=figure_dir, save_name="ModelDecomposition")
    

    
def plot_greens_function_animation(mesh, greens_function, point_func, frames=40, cmap='viridis', title="", save_dir = None, save_name = None, vmax=0.4):
    '''
    Plot an animation of the approximated Green's Function.

    Parameters:
        mesh (torch.Tensor)
        greens_function (Callable)
        point_func (Callable): Define which points to evaluate per frame.
    '''
    fig, ax = plt.subplots()
    ax.set_title(title)
    points = point_func(torch.linspace(1, frames, frames))
    sc = ax.scatter(mesh[0][:,0].detach().numpy(), mesh[0][:,1].detach().numpy(), 
                    c=greens_function(torch.zeros_like(mesh) + points[0], mesh).detach().numpy(), 
                    cmap=cmap)
    point_line = ax.scatter(*points[0], c="red")
    cbar = plt.colorbar(sc, ax=ax)  # create colorbar
    def update(frame):
        # for each frame, update the data stored on each artist.
        sc.set_array(greens_function(points[frame][None, None].expand(mesh.shape), mesh)[0].detach().numpy())
        point_line.set_offsets(points[frame])

    ani = animation.FuncAnimation(fig=fig, func=update, frames=frames, interval=30)
    plt.show()
    if save_dir is not None and save_name is not None:
        save_name = save_name + ".gif"
        ani.save(filename=save_dir + save_name, writer='pillow', fps=10, dpi=80)


from ngsolve import *
from expr_generation_utils import get_gaussian_expr, expr_to_func, func_input_wrapper
from plot_utils import plot_points, plot_multiple_points
import random, torch
from netgen.csg import *

ngsglobals.msg_level = 1

def generate_poisson_points(n: int, domain: tuple, eval_points: torch.Tensor, integration_points: torch.Tensor):
    '''
    Generates samples for the Poisson equation -∆u(x)=f(x) over Ω, u=0 over ∂Ω
      for n different Gaussian source terms evaluated each at eval_points.

    Parameters:
        n (int): number of samples to generate
        domain (tuple): (x_min, x_max, y_min, y_max) defining the domain
        eval_points (b x 2 torch.Tensor)
        integration_points (c x 2 torch.Tensor)
    '''

    if not (domain[0] == 0 and domain[1] == 1 and domain[2] == 0 and domain[3] == 1):
        raise NotImplementedError("Currently only implemented for a unit square mesh.")

    # generate a triangular mesh of mesh-size 0.2
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

    total_parameters = []
    total_eval_values = []
    total_integration_values = []

    for _ in range(n):
    # the right hand side
        f = LinearForm(fes)
        coeff_a = random.uniform(1, 10)
        sigma_x = random.uniform(0.01, 0.5)
        sigma_y = random.uniform(0.01, 0.5)
        mean_x = random.uniform(domain[0], domain[1])
        mean_y = random.uniform(domain[2], domain[3])
        parameters = {"coeff_a": coeff_a, "sigma_x": sigma_x, "sigma_y": sigma_y, "mean_x": mean_x, "mean_y": mean_y}

        cf = CoefficientFunction(coeff_a*exp(-((x-mean_x)**2 / (2*sigma_x**2) + (y-mean_y)**2 / (2*sigma_y**2))))
        f += cf * v * dx
        f.Assemble()

        gaussian_func = func_input_wrapper(expr_to_func(get_gaussian_expr(a=coeff_a, sigmas=(sigma_x, sigma_y), means=(mean_x, mean_y))))
        total_integration_values.append(gaussian_func(integration_points))

        # the solution field 
        gfu = GridFunction(fes)
        gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec

        values = []
        for point in eval_points:
            mesh_point = mesh(*point)
            values.append(gfu(mesh_point))

        total_parameters.append(parameters)
        total_eval_values.append(values)
    
    total_eval_values = torch.tensor(total_eval_values).flatten() # num_f_terms * len(eval_points) size Tensor 
    total_integration_values = torch.stack(total_integration_values) # num_f_terms x len(integration_points) size Tensor

    return {"parameters": total_parameters, "u_values": total_eval_values, "f_values": total_integration_values}


def generate_darcy_flow_points(n: int, domain: tuple, eval_points: torch.Tensor, integration_points: torch.Tensor, diffusion_gaussian_parameters=None):
    '''
    Generates samples for the Poisson equation ▽(a(x)▽u(x))=f(x) over Ω, u(x)=0 over ∂Ω
      for n different Gaussian source terms evaluated each at eval_points.

    Parameters:
        n (int): number of samples to generate
        domain (tuple): (x_min, x_max, y_min, y_max) defining the domain
        eval_points (b x 2 torch.Tensor)
        integration_points (c x 2 torch.Tensor)
    '''

    if not (domain[0] == 0 and domain[1] == 1 and domain[2] == 0 and domain[3] == 1):
        raise NotImplementedError("Currently only implemented for a unit square mesh.")

    # generate a triangular mesh of mesh-size 0.2
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))

    # H1-conforming finite element space
    fes = H1(mesh, order=3, dirichlet=[1,2,3,4])

    # define trial- and test-functions
    u = fes.TrialFunction()
    v = fes.TestFunction()

    # Define the diffusion term as a Gaussian function
    if diffusion_gaussian_parameters is None:
        coeff_a = random.uniform(1, 2)
        sigma_x = random.uniform(0.1, 0.3)
        sigma_y = random.uniform(0.1, 0.3)
        mean_x = random.uniform(domain[0], domain[1])
        mean_y = random.uniform(domain[2], domain[3])
        diffusion_parameters = {"coeff_a": coeff_a, "sigma_x": sigma_x, "sigma_y": sigma_y, "mean_x": mean_x, "mean_y": mean_y}
    else:
        coeff_a = diffusion_gaussian_parameters["coeff_a"]
        sigma_x = diffusion_gaussian_parameters["sigma_x"]
        sigma_y = diffusion_gaussian_parameters["sigma_y"]
        mean_x = diffusion_gaussian_parameters["mean_x"]
        mean_y = diffusion_gaussian_parameters["mean_y"]
        diffusion_parameters = diffusion_gaussian_parameters
    diffusion_term = CoefficientFunction(coeff_a*exp(-((x-mean_x)**2 / (2*sigma_x**2) + (y-mean_y)**2 / (2*sigma_y**2))))   
    # the bilinear-form 
    a = BilinearForm(fes, symmetric=True)
    a += -diffusion_term*grad(u)*grad(v)*dx
    a.Assemble()


    total_parameters = []
    total_eval_values = []
    total_integration_values = []

    diffusion_gaussian_func = func_input_wrapper(expr_to_func(get_gaussian_expr(a=coeff_a, sigmas=(sigma_x, sigma_y), means=(mean_x, mean_y))))
    diffusion_eval_point_values = diffusion_gaussian_func(eval_points)

    for _ in range(n):
    # the right hand side
        f = LinearForm(fes)
        coeff_a = random.uniform(1, 10)
        sigma_x = random.uniform(0.01, 0.5)
        sigma_y = random.uniform(0.01, 0.5)
        mean_x = random.uniform(domain[0], domain[1])
        mean_y = random.uniform(domain[2], domain[3])
        parameters = {"coeff_a": coeff_a, "sigma_x": sigma_x, "sigma_y": sigma_y, "mean_x": mean_x, "mean_y": mean_y}

        cf = CoefficientFunction(coeff_a*exp(-((x-mean_x)**2 / (2*sigma_x**2) + (y-mean_y)**2 / (2*sigma_y**2))))
        f += cf * v * dx
        f.Assemble()

        gaussian_func = func_input_wrapper(expr_to_func(get_gaussian_expr(a=coeff_a, sigmas=(sigma_x, sigma_y), means=(mean_x, mean_y))))
        total_integration_values.append(gaussian_func(integration_points))

        # the solution field 
        gfu = GridFunction(fes)
        gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec

        values = []
        for point in eval_points:
            mesh_point = mesh(*point)
            values.append(gfu(mesh_point))

        total_parameters.append(parameters)
        total_eval_values.append(values)
    
    total_eval_values = torch.tensor(total_eval_values).flatten() # num_f_terms * len(eval_points) size Tensor 
    if (total_eval_values == torch.inf).any() or (total_eval_values == -torch.inf).any():
        print("Warning: Inf values detected in total_eval_values, aborting.")
        assert False, "Inf values detected in total_eval_values, aborting."
    total_integration_values = torch.stack(total_integration_values) # num_f_terms x len(integration_points) size Tensor

    return {"parameters": total_parameters, 
            "u_values": total_eval_values, 
            "f_values": total_integration_values, 
            "diffusion_eval_point_values": diffusion_eval_point_values, 
            "diffusion_parameters": diffusion_parameters}


# mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))
# fes = H1(mesh, order=2, dirichlet=[1,2,3,4])

# coeff_a = random.uniform(1, 10)
# sigma_x = random.uniform(0.01, 0.5)
# sigma_y = random.uniform(0.01, 0.5)
# mean_x = random.uniform(0, 1)
# mean_y = random.uniform(0, 1)
# au = CoefficientFunction(coeff_a*exp(-((x-mean_x)**2 / (2*sigma_x**2) + (y-mean_y)**2 / (2*sigma_y**2))))

# Draw(au, mesh, "a(x)")
# u = fes.TrialFunction()  # symbolic object
# v = fes.TestFunction()   # symbolic object
# a = BilinearForm(-au*grad(u)*grad(v)*dx).Assemble()
# f = LinearForm(x*v*dx).Assemble()
# gfu = GridFunction(fes)  # solution
# gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec
# # plot the solution (netgen-gui only)
# Draw (gfu)
# # Draw (-grad(gfu), mesh, "Flux")


# mesh = sample_points(domain=(0,1,0,1), mesh_size=(20,20))
# return_dict = generate_darcy_flow_points(100, (0,1,0,1), mesh, mesh)

# plot_multiple_points([mesh, mesh, mesh, mesh, mesh], [return_dict["u_values"][0:400], return_dict["f_values"][0], return_dict["u_values"][400:800], return_dict["f_values"][1], return_dict["diffusion_eval_point_values"]])

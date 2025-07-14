# solve the Poisson equation -Delta u = f
# with Dirichlet boundary condition u = 0
from ngsolve import *
from expr_generation_utils import get_gaussian_expr, expr_to_func, func_input_wrapper
from plot_utils import plot_points, plot_multiple_points
import random, torch
from netgen.csg import *

ngsglobals.msg_level = 1

def generate_poisson_points(n: int, domain: tuple, eval_points: torch.Tensor, integration_points: torch.Tensor):
    '''
    Parameters:
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
    total_integration_values = torch.stack(total_integration_values)

    return {"parameters": total_parameters, "u_values": total_eval_values, "f_values": total_integration_values}



# plot the solution (netgen-gui only)
# Draw (gfu)
# Draw (-grad(gfu), mesh, "Flux")

# exact = 16*x*(1-x)*y*(1-y)
# print ("L2-error:", sqrt (Integrate ( (gfu-exact)*(gfu-exact), mesh)))

# mesh = sample_points((0,1,0,1), (20,20))
# return_dict = generate_poisson_points(100, (0,1,0,1), mesh, mesh)


# plot_multiple_points([mesh, mesh, mesh, mesh], [return_dict["total_eval_values"][0], return_dict["total_integration_values"][0], return_dict["total_eval_values"][1], return_dict["total_integration_values"][1],])

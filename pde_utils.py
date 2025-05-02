from typing import Tuple
import torch
from collections.abc import Callable
import scipy.integrate as integrate
import numpy as np
from data_generation import sample_mesh_points

def greens_function_poisson_eq_2d(coordinates: Tuple[float, float], center: Tuple[float, float]):
    """
    Greens function for the Poisson equation in 2D with Dirichlet boundary conditions.
    The function is defined as:
    G(x, y) = 1/(2*pi) * log(r), r < radius
    """
    r = ((coordinates[0] - center[0])**2 + (coordinates[1] - center[1])**2)
    G = -1/(2*np.pi) * np.log(r)
    return G

def test_source_term(x, y):
    """
    Test source term for the Poisson equation.
    """
    return np.sin(x) * np.cos(y)

def get_u_evaluation_func(greens_function: Callable[[Tuple[float, float], Tuple[float, float]], float], integrate_bool: bool = False):
    def evaluate_u_integrate(coordinate, domain: Tuple[float, float, float, float]):
        u = integrate.dblquad(
            lambda i, j: greens_function(coordinate, (i, j)) * test_source_term(i, j),
            domain[0], domain[1],
            lambda x: domain[2],
            lambda x: domain[3]
        )
        return u
    def evaluate_u_discrete(coordinate, domain: Tuple[float, float, float, float]):
        uniform_mesh=sample_mesh_points(domain[0], domain[1], domain[2], domain[3], 200, uniform=True)
        pred = torch.sum(greens_function(uniform_mesh,torch.zeros_like(uniform_mesh)+torch.tensor(coordinate)) * test_source_term(uniform_mesh[:,0],uniform_mesh[:,1]))
        return pred
    if integrate_bool:
        return evaluate_u_integrate
    else:
        return evaluate_u_discrete
    
def evaluate_model(model, coordinates, domain):
    evaluation_func = get_u_evaluation_func(model, False)
    evaluation = torch.zeros(len(coordinates))
    for i, coordinate in enumerate(coordinates):
        evaluation[i] = evaluation_func(coordinate=coordinate, domain=domain)
    return evaluation
    

# def evaluate_u(x, y, domain: Tuple[float, float, float, float]):
#     """
#     Evaluate the solution u at the given points (x, y) using the Green's function.
#     """
#     t1 = timer()
#     u = integrate.dblquad(
#         lambda i, j: greens_function_poisson_eq_2d((x, y), (i, j)) * test_source_term(i, j),
#         domain[0], domain[1],
#         lambda x: domain[2],
#         lambda x: domain[3]
#     )
#     t2 = timer()
#     print(t2-t1)
#     return u
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
    r = ((coordinates[:, 0] - center[:, 0])**2 + (coordinates[:, 1] - center[:, 1])**2)
    G = -1/(2*torch.pi) * torch.log(r)
    return G

def test_source_term(x, y, output_shape = None):
    """
    Test source term for the Poisson equation.
    """
    result = torch.sin(x) * torch.cos(y)
    if output_shape is None:
        return result
    return result.view(output_shape)

def get_u_evaluation_func(greens_function: Callable[[Tuple[float, float], Tuple[float, float]], float], integrate_bool: bool = False):
    def evaluate_u_integrate(coordinate, domain: Tuple[float, float, float, float]):
        u = integrate.dblquad(
            lambda i, j: greens_function(coordinate, (i, j)) * test_source_term(i, j),
            domain[0], domain[1],
            lambda x: domain[2],
            lambda x: domain[3]
        )
        return u[0]
    def evaluate_u_discrete(coordinate, domain: Tuple[float, float, float, float], coordinate_filter_radius = 1e-5):
        uniform_mesh=sample_mesh_points(domain[0], domain[1], domain[2], domain[3], 200, uniform=True)
        filter = torch.where((uniform_mesh - coordinate).pow(2).sum(1).sqrt() > coordinate_filter_radius)[0]
        filtered_mesh = uniform_mesh[filter]
        pred = torch.sum(greens_function(filtered_mesh,torch.zeros_like(filtered_mesh)+torch.tensor(coordinate)) * test_source_term(filtered_mesh[:,0],filtered_mesh[:,1]))
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


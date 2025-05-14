from typing import Tuple
import torch
from collections.abc import Callable
import scipy.integrate as integrate
import numpy as np
from data_generation_utils import sample_chebyshev_points, sample_uniform_mesh_points

# def greens_function_poisson_eq_2d(coordinates: Tuple[float, float], center: Tuple[float, float]):
#     """
#     Greens function for the Poisson equation in 2D with Dirichlet boundary conditions.
#     The function is defined as:
#     G(x, y) = 1/(2*pi) * log(r), r < radius
#     """
#     r = ((coordinates[:, 0] - center[:, 0])**2 + (coordinates[:, 1] - center[:, 1])**2)
#     G = -1/(2*torch.pi) * torch.log(r)
#     return G

def get_u_evaluation_func(greens_function: Callable[[Tuple[float, float], Tuple[float, float]], float], source_term: Callable[[Tuple[float, float], Tuple[float, float]], float], integrate_bool: bool = False, chebyshev: bool = False):
    '''
    Returns the function to evaluate the integration of the greens function.
    '''
    def evaluate_u_integrate(coordinate, domain: Tuple[float, float, float, float]):
        raise NotImplementedError
        u = integrate.dblquad(
            lambda i, j: greens_function(coordinate, (i, j)) * source_term(i, j),
            domain[0], domain[1],
            lambda x: domain[2],
            lambda x: domain[3]
        )
        return u[0]
    def evaluate_u_discrete(coordinate, domain: Tuple[float, float, float, float], coordinate_filter_radius = 1e-5):
        if chebyshev:
            raise NotImplementedError
            mesh=sample_chebyshev_points(domain, (20,20))
        else:
            mesh=sample_uniform_mesh_points(domain, (20, 20))
        filter = torch.where((mesh - coordinate).pow(2).sum(1).sqrt() > coordinate_filter_radius)[0]
        filtered_mesh = mesh[filter]
        area = (domain[1]-domain[0])*(domain[3]-domain[2])
        weights = torch.full((len(filtered_mesh),), area/len(filtered_mesh))
        greens_function_eval = greens_function(filtered_mesh,torch.zeros_like(filtered_mesh)+coordinate)
        source_term_eval = (source_term(filtered_mesh))
        pred = torch.sum(greens_function_eval*source_term_eval*weights)
        return pred
    
    def evaluate_u_chebyshev(coordinate, domain: Tuple[float, float, float, float], coordinate_filter_radius = 1e-5):
        raise NotImplementedError
        mesh=sample_chebyshev_points(domain, (20,20))
        filter = torch.where((mesh - coordinate).pow(2).sum(1).sqrt() > coordinate_filter_radius)[0]
        filtered_mesh = mesh[filter]
        greens_function_eval = greens_function(filtered_mesh,torch.zeros_like(filtered_mesh)+coordinate)
        source_term_eval = (source_term(filtered_mesh[:,0],filtered_mesh[:,1]))
        pred = torch.sum(greens_function_eval*source_term_eval)
        return pred
    if integrate_bool:
        return evaluate_u_integrate
    else:
        return evaluate_u_discrete
    
def evaluate_model(model, source_term, coordinates, domain, chebyshev: bool):
    '''
    Calculates the predicted values using the learned Green's Function model.
    '''
    evaluation_func = get_u_evaluation_func(greens_function=model, source_term=source_term, integrate_bool=False)
    evaluation = torch.zeros(len(coordinates))
    for i, coordinate in enumerate(coordinates):
        evaluation[i] = evaluation_func(coordinate=coordinate, domain=domain)
    return evaluation


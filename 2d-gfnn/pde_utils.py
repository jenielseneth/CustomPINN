from typing import Tuple
import torch
from collections.abc import Callable
import scipy.integrate as integrate
import numpy as np

def get_u_evaluation_func(greens_function: Callable[[Tuple[float, float], Tuple[float, float]], float], source_term: torch.Tensor, quadrature_mesh: torch.Tensor, chebyshev: bool = False):
    '''
    Returns the function to evaluate the integration of the greens function.
    '''
    def evaluate_u_discrete(coordinate, domain: Tuple[float, float, float, float], coordinate_filter_radius = 1e-5):
        filter = torch.where((quadrature_mesh - coordinate).pow(2).sum(1).sqrt() > coordinate_filter_radius)[0]
        filtered_mesh = quadrature_mesh[filter]
        source_term_eval = source_term[filter]
        area = (domain[1]-domain[0])*(domain[3]-domain[2])
        weights = torch.full((len(filtered_mesh),), area/len(filtered_mesh))
        greens_function_eval = greens_function(filtered_mesh,torch.zeros_like(filtered_mesh)+coordinate)
        pred = torch.sum(greens_function_eval*source_term_eval*weights)
        return pred
    
    def evaluate_u_chebyshev(coordinate, domain: Tuple[float, float, float, float], coordinate_filter_radius = 1e-5):
        raise NotImplementedError
    
    return evaluate_u_discrete
    
def evaluate_model(model, f_values, f_meshes, coordinates, f_inds, domain):
    '''
    Calculates the predicted values using the learned Green's Function model.
    '''
    evaluation = torch.zeros(len(coordinates))
    for i, coordinate in enumerate(coordinates):
        ind = f_inds[i]
        evaluation_func = get_u_evaluation_func(greens_function=model, source_term=f_values[ind], quadrature_mesh=f_meshes[ind])
        evaluation[i] = evaluation_func(coordinate=coordinate, domain=domain)
    return evaluation


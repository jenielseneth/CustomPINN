from typing import Tuple
import torch
from collections.abc import Callable
import scipy.integrate as integrate
import numpy as np

def get_u_evaluation_func(greens_function: Callable[[Tuple[float, float], Tuple[float, float]], float], source_term: torch.Tensor, quadrature_mesh: torch.Tensor):
    '''
    Returns the function to evaluate the integration of the greens function.
    '''
    def evaluate_u_discrete(coordinate, domain: Tuple[float, float, float, float], coordinate_filter_radius = 1e-5):
        filter = torch.where((quadrature_mesh - coordinate).pow(2).sum(1).sqrt() > coordinate_filter_radius)[0]
        filtered_mesh = quadrature_mesh[filter]
        source_term_eval = source_term[filter]
        area = (domain[1]-domain[0])*(domain[3]-domain[2])
        greens_function_eval = greens_function(torch.zeros_like(filtered_mesh)+coordinate, filtered_mesh)
        weights = torch.full((len(filtered_mesh),), area/len(filtered_mesh))
        pred = torch.sum(greens_function_eval*source_term_eval*weights)
        # pred = torch.sum(greens_function_eval*source_term_eval)
        return pred
    
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


##Updated pde_utils
# def eval_u_integral(greens_function: Callable[[Tuple[float, float], Tuple[float, float]], float], coordinate, domain: Tuple[float, float, float, float], f_mesh, f_values, coordinate_filter_radius = 1e-5):
#     '''
#     Assumes the model learns the quadrature weights.
#     coordinate: 1x2 Tensor
#     '''
#     filter = torch.where((f_mesh - coordinate).pow(2).sum(1).sqrt() > coordinate_filter_radius)[0]
#     filtered_mesh = f_mesh[filter]
#     source_term_eval = f_values[filter]
#     greens_function_eval = greens_function(torch.zeros_like(filtered_mesh)+coordinate, filtered_mesh)
#     pred = torch.sum(greens_function_eval*source_term_eval)
#     return pred

# def evaluate_model_2(model, f_values, f_mesh, coordinates, domain):
#     '''
#     Calculates the predicted values using the learned Green's Function model.
#     '''
#     evaluation = torch.zeros(len(coordinates))
#     for i, crd in enumerate(coordinates):
#         evaluation[i] = eval_u_integral(coordinate=crd, domain=domain, greens_function=model, f_values=f_values, f_mesh=f_mesh)
#     return evaluation

def eval_gf_integral(greens_function: Callable[[Tuple[float, float], Tuple[float, float]], float], coordinates, f_meshes, f_values, weights=None):

    '''
    Calculates the predicted values using the learned Green's Function model.
    This function evaluates the integral of the Green's function using a quadrature rule.
    coordinates: bx2 Tensor
    f_meshes: bxf_sizex2 Tensors, where f_size is the number of points on the source term mesh.
    f_values: bxf_size Tensor, where f_size is the number of points on the source term mesh.
    weights: f_size Tensor of weights for the quadrature rule, if None, we assume the model learns the weights.
    '''

    x_input = torch.zeros_like(f_meshes) + coordinates[:, None, :]  # b x f x 2 Tensor 
    y_input = f_meshes # b x f x 2 Tensor
    greens_function_eval = greens_function(x_input, y_input)
    integral = greens_function_eval*f_values  # b x f Tensor
    if weights is not None:
        integral = integral * weights[None, :]  # b x f Tensor, weights should be broadcasted
    pred = torch.sum(integral, -1)  # b Tensor, sum over the f dimension
    return pred

def bnd_eval_gf_integral(greens_function: Callable[[Tuple[float, float], Tuple[float, float]], float], coordinates, f_mesh, f_values, weights=None):
    '''
    This function is meant for data with separate boundary and collocation points.
    This function evaluates the integral of the Green's function using a quadrature rule.
    For a batch of coordinates on a single f_mesh. 
    Instead of filtering the mesh, we add slight noise to the coordinates so 
    coordinate: bx2 Tensor
    f_mesh: fx2 Tensor
    f_values: f Tensor
    weights: f Tensor of weights for the quadrature rule, if None, we assume the model learns the weights.
    '''
    bs, _ = coordinates.size()

    x_input = torch.zeros_like(f_mesh[None, :, :])+coordinates[:, None, :] #b x f x2 Tensor
    y_input = f_mesh[None, :, :].expand(bs, -1, -1) #b x f x 2 Tensor
    greens_function_eval = greens_function(x_input, y_input)# b x f Tensor
    integral = greens_function_eval*f_values # b x f Tensor
    if weights is not None:
        integral = integral * weights
    pred = torch.sum(integral, -1)
    return pred

# def bnd_eval_gf_integral_w_weights(greens_function: Callable[[Tuple[float, float], Tuple[float, float]], float], coordinates, f_mesh, f_values, weights):
#     '''
#     This function is meant for data with separate boundary and collocation points.
#     This function evaluates the integral of the Green's function using a quadrature rule with predetermined weights.
#     For a batch of coordinates on a single f_mesh.
#     Instead of filtering the mesh, we add slight noise to the coordinates so 
#     coordinate: bx2 Tensor
#     f_mesh: fx2 Tensor
#     f_values: f Tensor
#     weights: f Tensor of weights for the quadrature rule.
#     '''
#     bs, _ = coordinates.size()

#     x_input = torch.zeros_like(f_mesh[None, :, :])+coordinates[:, None, :] #b x f x2 Tensor
#     y_input = f_mesh[None, :, :].expand(bs, -1, -1) #b x f x 2 Tensor
#     greens_function_eval = greens_function(x_input, y_input) # b x f Tensor
#     integral = greens_function_eval*f_values*weights # b x f Tensor
#     pred = torch.sum(integral, -1)
#     return pred

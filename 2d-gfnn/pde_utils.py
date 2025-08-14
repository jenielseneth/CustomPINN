from typing import Tuple
import torch
from collections.abc import Callable
import scipy.integrate as integrate
import numpy as np
from dataset_utils import GreensConstantsDataclass, get_interior_boundary_idx
from data_generation_utils import gcd_chebyshev_mesh_size, sample_points
from chebyshev_utils import cheb_2d_impl
from constants_utils import Hyperparameters, mesh_type
from plot_utils import plot_points
from random_utils import resize_x_and_s
from loss import fetch_quadrature_weights
import time

class InferenceUtils:
    '''
    Experimental class to store features for training and inference.

    Attributes:
        chebyshev_evaluation_mesh (c x 2 Tensor): Tensor to store a chebyshev evaluation mesh for chebyshev inference.
        cheb_interior_indices (c_i Tensor): stores the indices of the interior points of the chebyshev evaluation mesh.
        cheb_boundary_indices (c_b Tensor): stores the indices of the boundary points of the chebyshev evaluation mesh.
        quadrature_weights (q Tensor): stores the quadrature weights of our datasets integration mesh (not to be confused with the chebyshev evaluation mesh).
    '''
    def __init__(self, constants: GreensConstantsDataclass, config: Hyperparameters):
        if not constants.evaluation_mesh_type == "chebyshev":
            
            chebyshev_mesh_size = gcd_chebyshev_mesh_size(constants.integration_mesh_sizes)
            self.chebyshev_evaluation_mesh = sample_points(constants.domain, chebyshev_mesh_size)
        else: 
            self.chebyshev_evaluation_mesh = sample_points(constants.domain, constants.evaluation_mesh_sizes)

        self.cheb_interior_indices, self.cheb_boundary_indices = get_interior_boundary_idx(domain=constants.domain, mesh=self.chebyshev_evaluation_mesh)

        if config.l_weights:
            self.quadrature_weights = None
        else:
            self.quadrature_weights = fetch_quadrature_weights(domain=constants.domain, 
                                                           integration_mesh_size=constants.integration_mesh_sizes, 
                                                           integration_mesh_type=constants.integration_mesh_type)
            self.quadrature_weights

    def to_device(self, device):
        self.quadrature_weights = self.quadrature_weights.to(device)
        self.chebyshev_evaluation_mesh = self.chebyshev_evaluation_mesh.to(device)

def u_laplacian_2d(greens_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 x: torch.Tensor, s: torch.Tensor, s_values: torch.Tensor, quadrature_weights: torch.Tensor):
    '''
    Calculates the 2D laplacian of u = ∫G(x,s)f(s)ds at points x with source points s. \n

    Parameters:
        greens_function (Callable): The Green's function to evaluate u with.
        x (torch.Tensor): b x 2 Tensor of points where the laplacian is evaluated.
        s (torch.Tensor): b x f x 2 | b x 2 Tensor of source points.
        s_values (torch.Tensor): f Tensor f(s) of source points s 
        quadrature_weights (torch.Tensor): f Tensor of quadrature weights for integration.
    
    Returns:
        lap (torch.Tensor): b x f Tensor returning the 2D Laplacian for every G(x,s) with respect to x.
    '''
    x.requires_grad = True
    s.requires_grad = True
    u = evaluate_greens_function_integral(greens_function=greens_function, 
                                          evaluation_mesh=x,
                                          integration_mesh=s,
                                          integration_mesh_values=s_values,
                                          quadrature_weights=quadrature_weights)
    grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    lap = 0.0
    for i in range(x.shape[-1]):
        grad2 = torch.autograd.grad(grad_u[..., i], x, grad_outputs=torch.ones_like(grad_u[..., i]), create_graph=True)[0][..., i]
        lap += grad2
    x.requires_grad = False
    s.requires_grad = False
    return lap

def greens_function_laplacian_2d(greens_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 x: torch.Tensor, s: torch.Tensor):
    '''
    Calculates the 2D laplacian of the Green's function at points x with source points s. \n

    Parameters:
        greens_function (Callable): The Green's function to evaluate.
        x (torch.Tensor): b x 2 Tensor of points where the laplacian is evaluated.
        s (torch.Tensor): f x 2 Tensor of source points.
    
    Returns:
        lap (torch.Tensor): b x f Tensor returning the 2D Laplacian for every G(x,s) with respect to x.
    '''
    x = x[:, None, :].expand(-1, s.shape[0], -1)  # b x f x 2 Tensor
    s = s[None, :, :].expand(x.shape[0], -1, -1)  # b x f x 2 Tensor
    x.requires_grad = True
    s.requires_grad = True
    g = greens_function(x, s)
    grad_g = torch.autograd.grad(g, x, grad_outputs=torch.ones_like(g), create_graph=True)[0]
    lap = 0.0
    for i in range(x.shape[-1]):
        grad2 = torch.autograd.grad(grad_g[..., i], x, grad_outputs=torch.ones_like(grad_g[..., i]), create_graph=True)[0][..., i]
        lap += grad2
    x.requires_grad = False
    s.requires_grad = False
    return lap

def greens_function_darcy_flow_operator_2d(greens_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 x: torch.Tensor, s: torch.Tensor):
    '''
    Calculates the 2D laplacian of the Green's function at points x with source points s. \n

    Parameters:
        greens_function (Callable): The Green's function to evaluate.
        x (torch.Tensor): b x 2 Tensor of points where the laplacian is evaluated.
        s (torch.Tensor): f x 2 Tensor of source points.
    
    Returns:
        lap (torch.Tensor): b x f Tensor returning the 2D Laplacian for every G(x,s) with respect to x.
    '''
    x = x[:, None, :].expand(-1, s.shape[0], -1)  # b x f x 2 Tensor
    s = s[None, :, :].expand(x.shape[0], -1, -1)  # b x f x 2 Tensor
    x.requires_grad = True
    s.requires_grad = True
    g = greens_function(x, s)
    grad_g = torch.autograd.grad(g, x, grad_outputs=torch.ones_like(g), create_graph=True)[0]
    lap = 0.0
    for i in range(x.shape[-1]):
        grad2 = torch.autograd.grad(grad_g[..., i], x, grad_outputs=torch.ones_like(grad_g[..., i]), create_graph=True)[0][..., i]
        lap += grad2
    x.requires_grad = False
    s.requires_grad = False
    return lap

def evaluate_greens_function_integral(greens_function: Callable[[Tuple[float, float], Tuple[float, float]], float], 
                                      evaluation_mesh: torch.Tensor, 
                                      integration_mesh: torch.Tensor,
                                      integration_mesh_values: torch.Tensor,
                                      quadrature_weights: torch.Tensor):

    '''
    Calculates the predicted values using the learned Green's Function model. \n
    This function evaluates the integral of the Green's function using a quadrature rule. \n
    :param Tensor evaluation_mesh: b x 2 Tensor
    :param Tensor integration_mesh: b x f_size x 2 Tensor | f_size x 2 Tensor, where f_size is the number of points on the source term mesh.
    :param Tensor integration_mesh_values: b x f_size Tensor | f_size Tensor, where f_size is the number of points on the source term mesh.
    :param Tensor quadrature_weights: f_size Tensor of weights for the quadrature rule, if None, we assume the model learns the weights.

    :return Tensor pred: b Tensor.
    '''

    # Establish evaluation_mesh device as main device.
    assert evaluation_mesh.device == integration_mesh_values.device, f"evaluation_mesh ({evaluation_mesh.device}) and integration_mesh_values ({integration_mesh_values.device}) are not on the same device. "
    assert evaluation_mesh.device == integration_mesh.device, f"evaluation_mesh ({evaluation_mesh.device}) and dataset_constants ({integration_mesh.device}) are not on the same device. "
    assert evaluation_mesh.device == quadrature_weights.device, f"evaluation_mesh ({evaluation_mesh.device}) and quadrature_weights ({quadrature_weights.device}) are not on the same device. "
    
    # integration_mesh = dataset_constants.integration_mesh
    weights = quadrature_weights

    assert evaluation_mesh.dim() == 2 and evaluation_mesh.shape[1] == 2, f"evaluation_mesh ({evaluation_mesh.shape}) must be a b x 2 Tensor."

    if integration_mesh.dim() == 3:
        assert evaluation_mesh.shape[0] == integration_mesh.shape[0], f"integration_mesh ({integration_mesh.shape}) must either have the same size in dim 0 as evaluation_mesh ({evaluation_mesh.shape}), or have the size: f_size x 2 Tensor."

    elif integration_mesh.dim() == 2:
        integration_mesh = integration_mesh[None, :, :].expand(evaluation_mesh.shape[0], -1, -1)
    else: 
        raise ValueError("integration_mesh must be of either dimension 2 or 3.")
    
    if integration_mesh_values.dim() == 2:
        assert evaluation_mesh.shape[0] == integration_mesh_values.shape[0], f"integration_mesh_values with shape {integration_mesh_values.shape} must either have the same size in dim 0 as evaluation_mesh, or have the size: f_size Tensor."
    elif integration_mesh_values.dim() == 1:
        assert integration_mesh_values.shape[0] == integration_mesh.shape[1], f"integration_mesh_values with shape {integration_mesh_values.shape} must have the same size in dim 0 as integration_mesh ({integration_mesh.shape})."
        integration_mesh_values = integration_mesh_values[None, :].expand(evaluation_mesh.shape[0], -1)
    else:
        raise ValueError("integration_mesh_values must be of either dimension 1 or 2.")
    
    assert quadrature_weights.dim() == 1 and quadrature_weights.shape[0] == integration_mesh_values.shape[1], f"quadrature_weights {quadrature_weights.shape}) must be a f_size Tensor."

    x_input = evaluation_mesh[:, None, :].expand(-1, integration_mesh.shape[1], -1)  # b x f x 2 Tensor 
    s_input = integration_mesh # b x f x 2 Tensor

    assert x_input.shape == s_input.shape and x_input.dim() == s_input.dim() == 3

    greens_function_eval = greens_function(x_input, s_input)
    integral = greens_function_eval*integration_mesh_values  # b x f Tensor
    if weights is not None:
        integral = integral * weights[None, :]  # b x f Tensor, weights should be broadcasted
        assert integral.shape == (evaluation_mesh.shape[0], integration_mesh_values.shape[1]), f"integral shape {integral.shape} does not match expected shape ({evaluation_mesh.shape[0]}, {integration_mesh_values.shape[1]})."
    pred = torch.sum(integral, -1)  # b Tensor, sum over the f dimension
    return pred


def chebyshev_inference(greens_function: Callable[[Tuple[float, float], Tuple[float, float]], float], 
                        evaluation_coordinates: torch.Tensor, integration_mesh_values: torch.Tensor, 
                        dataset_constants: GreensConstantsDataclass, inference_utils: InferenceUtils,
                        boundary_condition: float):
    '''
    Calculates u(x) on an evaluation chebyshev mesh, which is used to interpolate the values at evaluation_coordinates. \n

    :param b x 2 Tensor evaluation_coordinates: 
    :param f integration_mesh_values: 
    '''
    assert type(boundary_condition) == float, "Currently only implemented for constant boundary conditions." 
    
    u_pred_cheb = evaluate_greens_function_integral(greens_function=greens_function, 
                                        evaluation_mesh=inference_utils.chebyshev_evaluation_mesh,
                                        integration_mesh_values=integration_mesh_values,
                                        integration_mesh=dataset_constants.integration_mesh, 
                                        quadrature_weights=inference_utils.quadrature_weights)
    u_pred_cheb[inference_utils.cheb_boundary_indices] = boundary_condition # Boundary condition
    u_pred_eval = cheb_2d_impl(eval_points=evaluation_coordinates, chebyshev_values=u_pred_cheb, 
                               chebyshev_size=dataset_constants.evaluation_mesh_sizes, domain=dataset_constants.domain)

    return u_pred_eval
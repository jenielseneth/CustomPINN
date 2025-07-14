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
            
            chebyshev_mesh_size = gcd_chebyshev_mesh_size(constants.integration_mesh_size)
            self.chebyshev_evaluation_mesh = sample_points(constants.domain, chebyshev_mesh_size)
        else: 
            self.chebyshev_evaluation_mesh = sample_points(constants.domain, constants.evaluation_mesh_size)

        self.cheb_interior_indices, self.cheb_boundary_indices = get_interior_boundary_idx(domain=constants.domain, mesh=self.chebyshev_evaluation_mesh)

        if config.l_weights:
            self.quadrature_weights = None
        else:
            self.quadrature_weights = fetch_quadrature_weights(domain=constants.domain, 
                                                           integration_mesh_size=constants.integration_mesh_size, 
                                                           integration_mesh_type=constants.integration_mesh_type)
            self.quadrature_weights

    def to_device(self, device):
        self.quadrature_weights = self.quadrature_weights.to(device)
        self.chebyshev_evaluation_mesh = self.chebyshev_evaluation_mesh.to(device)


def greens_function_laplacian_2d(greens_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 delta_function_center: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    '''
    Computes the Laplacian of a 2D function using auto-diff.
    '''
    def f_jacobian_x(x, y):
        return torch.diag(torch.func.jacrev(greens_function, argnums=0)(x, y))

    def f_jacobian_y(x, y):
        return torch.diag(torch.func.jacrev(greens_function, argnums=1)(x, y))

    start = time.time()
    hessian_xx = torch.diag(torch.func.jacfwd(f_jacobian_x, argnums=0)(delta_function_center, y))
    hessian_yy = torch.diag(torch.func.jacfwd(f_jacobian_y, argnums=1)(delta_function_center, y))
    hessian = torch.vstack((hessian_xx, hessian_yy)).mT
    end = time.time()
    print(f"Time taken for Hessian calculation: {end - start} seconds")
    return hessian.sum(dim=-1)  # Return the sum of the diagonal elements (Laplacian)


def evaluate_greens_function_integral(greens_function: Callable[[Tuple[float, float], Tuple[float, float]], float], 
                                      evaluation_mesh: torch.Tensor, integration_mesh_values: torch.Tensor, 
                                      integration_mesh: torch.Tensor, quadrature_weights: torch.Tensor):

    '''
    Calculates the predicted values using the learned Green's Function model. \n
    This function evaluates the integral of the Green's function using a quadrature rule. \n
    :param Tensor evaluation_mesh: b x 2 Tensor
    :param Tensor integration_mesh: b x f_size x 2 Tensor | f_size x 2 Tensor, where f_size is the number of points on the source term mesh.
    :param Tensor integration_mesh_values: b x f_size Tensor | f_size Tensor, where f_size is the number of points on the source term mesh.
    :param Tensor weights: f_size Tensor of weights for the quadrature rule, if None, we assume the model learns the weights.

    :return Tensor pred: b Tensor.
    '''

    # Establish evaluation_mesh device as main device.
    assert evaluation_mesh.device == integration_mesh_values.device, f"evaluation_mesh ({evaluation_mesh.device}) and integration_mesh_values ({integration_mesh_values.device}) are not on the same device. "
    assert evaluation_mesh.device == integration_mesh.device, f"evaluation_mesh ({evaluation_mesh.device}) and dataset_constants ({integration_mesh.device}) are not on the same device. "
    assert evaluation_mesh.device == quadrature_weights.device, f"evaluation_mesh ({evaluation_mesh.device}) and quadrature_weights ({quadrature_weights.device}) are not on the same device. "
    
    # integration_mesh = dataset_constants.integration_mesh
    weights = quadrature_weights

    assert evaluation_mesh.dim() == 2 and evaluation_mesh.shape[1] == 2, "evaluation_mesh must be a b x 2 Tensor."

    if integration_mesh.dim() == 3:
        assert evaluation_mesh.shape[0] == integration_mesh.shape[0], f"integration_mesh ({integration_mesh.shape}) must either have the same size in dim 0 as evaluation_mesh ({evaluation_mesh.shape}), or have the size: f_size x 2 Tensor."

    elif integration_mesh.dim() == 2:
        integration_mesh = integration_mesh[None, :, :].expand(evaluation_mesh.shape[0], -1, -1)
    else: 
        raise ValueError("integration_mesh must be of either dimension 2 or 3.")
    
    if integration_mesh_values.dim() == 2:
        assert evaluation_mesh.shape[0] == integration_mesh_values.shape[0], f"integration_mesh_values with shape {integration_mesh_values.shape} must either have the same size in dim 0 as evaluation_mesh, or have the size: f_size Tensor."
    elif integration_mesh_values.dim() == 1:
        integration_mesh_values = integration_mesh_values[None, :].expand(evaluation_mesh.shape[0], -1)
    else:
        raise ValueError("integration_mesh_values must be of either dimension 1 or 2.")
    
    assert quadrature_weights.dim() == 1 and quadrature_weights.shape[0] == integration_mesh_values.shape[1], f"quadrature_weights {quadrature_weights.shape}) must be a f_size Tensor."

    x_input = evaluation_mesh[:, None, :].expand(-1, integration_mesh.shape[1], -1)  # b x f x 2 Tensor 
    y_input = integration_mesh # b x f x 2 Tensor

    assert x_input.shape == y_input.shape and x_input.dim() == y_input.dim() == 3

    greens_function_eval = greens_function(x_input, y_input)
    integral = greens_function_eval*integration_mesh_values  # b x f Tensor
    if weights is not None:
        integral = integral * weights[None, :]  # b x f Tensor, weights should be broadcasted
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
                               chebyshev_size=dataset_constants.evaluation_mesh_size, domain=dataset_constants.domain)

    return u_pred_eval
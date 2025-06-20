from typing import Tuple
import torch
from collections.abc import Callable
import scipy.integrate as integrate
import numpy as np
from dataset_utils import GreensConstantsDataclass
from chebyshev_utils import cheb_2d_impl_2, clenshaw_curtis_weights_2d
from constants_utils import mesh_type
from loss import fetch_quadrature_weights

    

def evaluate_greens_function_integral(greens_function: Callable[[Tuple[float, float], Tuple[float, float]], float], evaluation_mesh, integration_mesh_values, dataset_constants: GreensConstantsDataclass, weights=None):

    '''
    Calculates the predicted values using the learned Green's Function model. \n
    This function evaluates the integral of the Green's function using a quadrature rule. \n
    :param Tensor evaluation_mesh: b x 2 Tensor
    :param Tensor integration_mesh: b x f_size x 2 Tensor | f_size x 2 Tensor, where f_size is the number of points on the source term mesh.
    :param Tensor integration_mesh_values: b x f_size Tensor | f_size Tensor, where f_size is the number of points on the source term mesh.
    :param Tensor weights: f_size Tensor of weights for the quadrature rule, if None, we assume the model learns the weights.
    '''
    integration_mesh = dataset_constants.integration_mesh
    weights = dataset_constants.quadrature_weights

    assert evaluation_mesh.dim() == 2 and evaluation_mesh.shape[1] == 2, "evaluation_mesh must be a b x 2 Tensor."

    if integration_mesh.dim() == 3:
        assert evaluation_mesh.shape[0] == integration_mesh.shape[0], "integration_mesh must either have the same size in dim 0 as evaluation_mesh, or have the size: f_size x 2 Tensor."

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

    x_input = torch.zeros_like(integration_mesh) + evaluation_mesh[:, None, :]  # b x f x 2 Tensor 
    y_input = integration_mesh # b x f x 2 Tensor

    assert x_input.shape == y_input.shape and x_input.dim() == y_input.dim() == 3

    greens_function_eval = greens_function(x_input, y_input)
    integral = greens_function_eval*integration_mesh_values  # b x f Tensor
    if weights is not None:
        integral = integral * weights[None, :]  # b x f Tensor, weights should be broadcasted
    pred = torch.sum(integral, -1)  # b Tensor, sum over the f dimension
    return pred

def chebyshev_inference(greens_function: Callable[[Tuple[float, float], Tuple[float, float]], float], evaluation_coordinates, integration_mesh_values, dataset_constants: GreensConstantsDataclass, l_weights: bool):
    '''
    Calculates u(x) on an evaluation chebyshev mesh, which is then used to 
    '''
    if l_weights:
        weights = None
    else:
        weights = dataset_constants.quadrature_weights
    
    u_pred = evaluate_greens_function_integral(greens_function=greens_function, 
                                        evaluation_mesh=dataset_constants.chebyshev_evaluation_mesh,
                                        dataset_constants=dataset_constants,
                                        integration_mesh=dataset_constants.integration_mesh, integration_mesh_values=integration_mesh_values, weights=weights)
    # u_pred[bnd] = 0 # Boundary condition
    u_pred_uniform = cheb_2d_impl_2(eval_points=evaluation_coordinates, values=u_pred, chebyshev_size=dataset_constants.evaluation_mesh_size, domain=dataset_constants.domain)


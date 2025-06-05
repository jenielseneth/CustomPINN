import torch
from tqdm import tqdm

from plot_utils import plot_multiple_points, plot_points
from pde_utils import bnd_eval_gf_integral, eval_gf_integral
from chebyshev_utils import clenshaw_curtis_weights_2d

# class DataPredLoss(object):
#     def __init__(self, num_eval_points):
#         super().__init__()
#         self.num_eval_points = num_eval_points

#     def __call__(self, greens_function_approx, f_values, f_meshes, coordinates, domain, f_inds, u, *args, **kwargs):
#         u_pred = evaluate_model(model=greens_function_approx, f_values=f_values, f_meshes=f_meshes, f_inds=f_inds, coordinates=coordinates, domain=domain)
#         diff = torch.nn.functional.mse_loss(u_pred, u)
#         # plot_multiple_points([coordinates, coordinates], values_list=[u_pred, u], title_list=["Predicted Values", "Ground Truth"])
#         return diff
    
class DataPredLoss(object):
    '''
    Used for datasets with joint collocation and boundary points.
    This loss function evaluates the integral of the Green's function using a quadrature rule.
    Here, we use MultiDatasetWrapper which maps the respective source term mesh and values to a single point we want
        to evaluate u(x) at.
    '''
    def __init__(self, domain, num_points, l_weights: bool):
        super().__init__()
        area_ratio = (domain[1]-domain[0])*(domain[3]-domain[2])/(4)
        x_num, y_num = num_points
        self.weights = clenshaw_curtis_weights_2d((x_num-1, y_num-1)) * area_ratio
        self.l_weights = l_weights

    def __call__(self, greens_function_approx, f_meshes_batch, f_values_batch, coordinates_batch, u_batch, *args, **kwargs):
        '''
        greens_function_approx: Callable that approximates the Green's function, takes in coordinates and returns a tensor of values
        f_meshes_batch: Tensor of source term meshes: b x num_f x 2 Tensor, where b is the batch size and num_f is the number of source term points
        f_values_batch: Tensor of source term values: b x num_f Tensor, where b is the batch size and num_f is the number of source term points
        coordinates_batch: Tensor of coordinates to evaluate the Green's function at: b x 2 Tensor
        u_batch: Tensor of ground truth values at the coordinates: b x 1 Tensor
        '''
        if len(coordinates_batch) == 0:
            loss = 0
        else:
            if self.l_weights:
                u_pred = eval_gf_integral(greens_function=greens_function_approx, f_values=f_values_batch, f_meshes=f_meshes_batch, coordinates=coordinates_batch)
            else:
                u_pred = eval_gf_integral(greens_function=greens_function_approx, f_values=f_values_batch, f_meshes=f_meshes_batch, coordinates=coordinates_batch, weights=self.weights)
            loss = torch.nn.functional.mse_loss(u_pred, u_batch)
        # plot_multiple_points([coordinates_batch, coordinates_batch], values_list=[u_pred, u_batch], title_list=["Predicted Collocation Values", "Ground Collocation Truth"])
        return loss
    

class BndDataPredLoss(object):
    '''
    Used for datasets with separate collocation and boundary points.
    This loss function evaluates the integral of the Green's function using a quadrature rule.
    Since incoorporating collocation and boundary points into a single loss function requires the entire mesh points
        over the domain, we cannot arbitrarily shuffle the data over the different datasets. 
        Instead, we must input at each loss calculation the entire dataset of collocation points and boundary points 
            corresponding to one source term.
    '''
    def __init__(self, domain, num_points, l_weights: bool):
        super().__init__()
        area_ratio = (domain[1]-domain[0])*(domain[3]-domain[2])/(4)
        x_num, y_num = num_points
        self.weights = clenshaw_curtis_weights_2d((x_num-1, y_num-1)) * area_ratio #We assume the weights are clenshaw-curtis weights
        self.l_weights = l_weights

    def __call__(self, greens_function_approx, f_values_batch, f_mesh_batch, coordinates_batch, u_batch, *args, **kwargs):
        total_loss = 0
        for i, coordinates in enumerate(coordinates_batch):
            if len(coordinates) == 0:
                loss = 0
            else:
                if self.l_weights:
                    u_pred = bnd_eval_gf_integral(greens_function=greens_function_approx, f_values=f_values_batch[i], f_mesh=f_mesh_batch[i], coordinates=coordinates)
                else:
                    u_pred = bnd_eval_gf_integral(greens_function=greens_function_approx, f_values=f_values_batch[i], f_mesh=f_mesh_batch[i], coordinates=coordinates, weights=self.weights)
                loss = torch.nn.functional.mse_loss(u_pred, u_batch[i])
            # plot_multiple_points([coordinates, coordinates], values_list=[u_pred, u_batch[i]], title_list=["Predicted Collocation Values", "Ground Collocation Truth"])
            total_loss += loss
        return total_loss
    
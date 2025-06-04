import torch
from tqdm import tqdm

from plot_utils import plot_multiple_points, plot_points
from pde_utils import eval_u_integral_2, eval_u_integral_3
from chebyshev import clenshaw_curtis_weights_2d

# class DataPredLoss(object):
#     def __init__(self, num_eval_points):
#         super().__init__()
#         self.num_eval_points = num_eval_points

#     def __call__(self, greens_function_approx, f_values, f_meshes, coordinates, domain, f_inds, u, *args, **kwargs):
#         u_pred = evaluate_model(model=greens_function_approx, f_values=f_values, f_meshes=f_meshes, f_inds=f_inds, coordinates=coordinates, domain=domain)
#         diff = torch.nn.functional.mse_loss(u_pred, u)
#         # plot_multiple_points([coordinates, coordinates], values_list=[u_pred, u], title_list=["Predicted Values", "Ground Truth"])
#         return diff
    
class UpdatedDataPredLoss(object):
    def __init__(self, domain, num_points):
        super().__init__()
        area_ratio = (domain[1]-domain[0])*(domain[3]-domain[2])/(4)
        x_num, y_num = num_points
        self.weights = clenshaw_curtis_weights_2d((x_num-1, y_num-1)) * area_ratio

    def __call__(self, greens_function_approx, f_values_batch, f_mesh_batch, coordinates_batch, u_batch, domain, *args, **kwargs):
        total_loss = 0
        for i, coordinates in enumerate(coordinates_batch):
            if len(coordinates) == 0:
                loss = 0
            else:
                # u_pred = eval_u_integral_2(greens_function=greens_function_approx, f_values=f_values_batch[i], f_mesh=f_mesh_batch[i], coordinates=coordinates)
                u_pred = eval_u_integral_3(greens_function=greens_function_approx, f_values=f_values_batch[i], f_mesh=f_mesh_batch[i], coordinates=coordinates, weights=self.weights)
                loss = torch.nn.functional.mse_loss(u_pred, u_batch[i])
            # plot_multiple_points([coordinates, coordinates], values_list=[u_pred, u_batch[i]], title_list=["Predicted Collocation Values", "Ground Collocation Truth"])
            total_loss += loss
        return total_loss
    
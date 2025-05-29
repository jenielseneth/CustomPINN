import torch
from tqdm import tqdm

from plot_utils import plot_multiple_points, plot_points
from pde_utils import eval_u_integral_2

class relMSELoss(object):
    def __init__(self):
        super().__init__()
    
    def __call__(self, y_pred, y, *args, **kwds):
        """
        Parameters
        ----------
        y_pred : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        """
        diff = torch.nn.functional.mse_loss(y_pred, y, reduction="none")
        
        ynorm = torch.nn.functional.mse_loss(y, torch.zeros_like(y), reduction="none")
        ynorm = torch.where(ynorm==0, 1, ynorm)
        loss = diff/ynorm
        return torch.sum(loss)

def print_graph(tensor, indent=0):
    if not hasattr(tensor, "grad_fn"):
        print(" " * indent + f"Leaf tensor: {tensor}")
        return
    print(" " * indent + f"grad_fn: {tensor.grad_fn}")
    for fn, _ in tensor.grad_fn.next_functions:
        if fn is not None:
            print_graph(fn, indent + 4)


def discrete_integration(coordinate, mesh, greens_function_approx, f_source_term, coordinate_filter_radius = 1e-5):
        filter = torch.where((mesh - coordinate).pow(2).sum(1).sqrt() > coordinate_filter_radius)[0]
        filtered_mesh = mesh[filter]
        source = f_source_term(filtered_mesh[:,0],filtered_mesh[:,1])
        pred = torch.sum(greens_function_approx(filtered_mesh, torch.zeros_like(filtered_mesh)+torch.tensor(coordinate)) * source)
        return pred

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
    def __init__(self):
        super().__init__()

    def __call__(self, greens_function_approx, f_values_batch, f_mesh_batch, coordinates_batch, u_batch, domain, *args, **kwargs):
        total_loss = 0
        for i, coordinates in enumerate(coordinates_batch):
            if len(coordinates) == 0:
                loss = 0
            else:
                u_pred = eval_u_integral_2(greens_function=greens_function_approx, f_values=f_values_batch[i], f_mesh=f_mesh_batch[i], coordinates=coordinates)
                loss = torch.nn.functional.mse_loss(u_pred, u_batch[i])
            # plot_multiple_points([coordinates, coordinates], values_list=[u_pred, u_batch[i]], title_list=["Predicted Collocation Values", "Ground Collocation Truth"])
            total_loss += loss
        return total_loss
    
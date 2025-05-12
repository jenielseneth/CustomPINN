import torch
from tqdm import tqdm

from chebyshev import plot_multiple_points, plot_points
from data_generation_utils import sample_mesh_points
from torchviz import make_dot
from pde_utils import get_u_evaluation_func

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

class CustomDataPredLoss(object):
    def __init__(self, num_eval_points):
        super().__init__()
        self.num_eval_points = num_eval_points

    def __call__(self, greens_function_approx, f_source_term, coordinates, domain, u, *args, **kwargs):
        u_pred = torch.zeros_like(u)
        uniform_mesh=sample_mesh_points(domain, self.num_eval_points, uniform=True)
        for c in range(len(coordinates)):
            filter = torch.where((uniform_mesh - coordinates[c]).pow(2).sum(1).sqrt() > 1e-5)[0]
            filtered_mesh = uniform_mesh[filter]
            source = f_source_term(filtered_mesh[:,0],filtered_mesh[:,1])
            pred = torch.sum(greens_function_approx(filtered_mesh, torch.zeros_like(filtered_mesh)+coordinates[c]) * source)
            u_pred[c] = pred
        diff = torch.nn.functional.mse_loss(u_pred, u)
        return diff / len(u_pred)
    

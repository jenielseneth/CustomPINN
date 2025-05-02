import torch
from tqdm import tqdm

from data_generation import sample_mesh_points

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
        diff = torch.nn.functional.mse_loss(y_pred, y)
        ynorm = torch.nn.functional.mse_loss(y, torch.zeros_like(y))
        if ynorm == 0:
            return diff
        else:
            return diff/ynorm

class CustomLoss(object):
    def __init__(self):
        super().__init__()
    
    def __call__(self, greens_function_approx, f_source_term, coordinate, domain, u, *args, **kwargs):
        u_pred = torch.zeros_like(u)
        uniform_mesh=sample_mesh_points(domain[0], domain[1], domain[2], domain[3], 200, uniform=True)
        # print(greens_function_approx(coordinate,torch.zeros_like(coordinate)+torch.tensor((s,t))) * f_source_term(s,t))
        for c in range(len(coordinate)):
            mesh_filtered = torch.where((uniform_mesh - coordinate[c]).pow(2).sum(1).sqrt() > 1e-5)
            pred = torch.sum(greens_function_approx(uniform_mesh[mesh_filtered],torch.zeros_like(uniform_mesh[mesh_filtered])+torch.tensor(coordinate[c])) * f_source_term(uniform_mesh[mesh_filtered][:,0],uniform_mesh[mesh_filtered][:,1]))
            u_pred[c] = pred
            
            
        diff = torch.nn.functional.mse_loss(u_pred, u)
        return diff
    
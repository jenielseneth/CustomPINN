import torch
from tqdm import tqdm
from torch.nn.modules.loss import _Loss

from plot_utils import plot_multiple_points, plot_points
from chebyshev_utils import clenshaw_curtis_weights_2d
from constants_utils import mesh_type


def fetch_quadrature_weights(domain, integration_mesh_size, integration_mesh_type: mesh_type):
    '''
    Returns quadrature weights for the corresponding data mesh type.

    :param tuple domain: (x_min, x_max, y_min, y_max) tuple.
    :param tuple mesh_size: (x_num, y_num) number of points per axis over mesh.
    :param integration_mesh_type: Type of mesh we integrate over.

    '''
    if integration_mesh_type == "chebyshev":
        area_ratio = (domain[1]-domain[0])*(domain[3]-domain[2])/(4)
        x_num, y_num = integration_mesh_size
        weights = clenshaw_curtis_weights_2d((x_num-1, y_num-1)) * area_ratio

    elif integration_mesh_type == "uniform":
        area = (domain[1]-domain[0])*(domain[3]-domain[2])
        x_num, y_num = integration_mesh_size
        weights = torch.full((x_num * y_num,), area/(x_num*y_num))

    elif integration_mesh_type == "random":
        assert False, "Random mesh type is not supported for quadrature weights."

    else:
        raise ValueError(f"Unknown mesh type: {integration_mesh_type}. Supported types are 'chebyshev', 'uniform', and 'random'.")
    
    return weights

class MAPELoss(_Loss):
    def __init__(self, eps=1e-8, reduction='mean'):
        super(MAPELoss, self).__init__(reduction=reduction)
        self.eps = eps  # Prevent division by zero

    def forward(self, y_pred, y_true):
        y_pred = y_pred.float()
        y_true = y_true.float()

        # Calculate element-wise absolute percentage error
        ape = torch.abs((y_true - y_pred) / (y_true + self.eps))

        if self.reduction == 'mean':
            return torch.mean(ape) * 100
        elif self.reduction == 'sum':
            return torch.sum(ape) * 100
        else:  # 'none'
            return ape * 100
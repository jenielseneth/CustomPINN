import math
import numpy as np
import torch

from data_generation_utils import sample_points
from plot_utils import plot_convergence_rate, plot_points
from scipy.stats import linregress

def sample_chebyshev_points_2(domain, num_points: tuple):
    x_num, y_num = num_points
    x_min, x_max, y_min, y_max = domain
    points_x = torch.linspace(0, x_num-1, x_num) * torch.pi / (x_num-1)
    points_x = torch.cos(points_x)
    points_y = torch.linspace(0, y_num-1, y_num) * torch.pi / (y_num-1)
    points_y = torch.cos(points_y)
    points_x += 1
    points_x /= 2
    points_y += 1
    points_y /= 2
    points_x = points_x * (x_max-x_min) + x_min
    points_y = points_y * (y_max-y_min) + y_min
    xy = torch.zeros((x_num, y_num, 2))
    for i in range(x_num):
        for j in range(y_num):
            xy[i,j] = torch.tensor([points_x[i], points_y[j]])
    return xy

def cheb_1d_points(domain, num_points: int) -> torch.Tensor:
    x_min, x_max = domain
    points_x = torch.linspace(0, num_points-1, num_points) * torch.pi / (num_points-1)
    points_x = torch.cos(points_x)
    points_x += 1
    points_x /= 2
    points_x = points_x * (x_max-x_min) + x_min
    return points_x


def cheb_weights(n) -> torch.Tensor:
    '''
    Calculates the Chebyshev weights for 0,1,...,n points of the second kind.
    If you have e.g. a total length 10 chebyshev nodes, input n = 9 to get the weights for nodes 0,1,...,9. \n

    Parameters 
    ----------------
    n: number of nodes
    '''
    weights = torch.linspace(0, n, n+1)
    weights = torch.pow(-1, weights)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return weights

def cheb_1d_impl(eval_points: torch.Tensor, values: torch.Tensor, domain: tuple) -> torch.Tensor:
    '''
    Chebyshev interpolation in 1D. \n
    :param Tensor eval_points: m x 1 (m points to evaluate) 
    :param Tensor values: n x 1 (n values at the Chebyshev nodes)
    :param tuple domain: (x_min, x_max)

    :return Tensor eval: m x 1 (m values at eval_points) 
    '''

    n = len(values)
    points = cheb_1d_points(domain, n)
    weights = cheb_weights(n-1)
    eval = torch.zeros_like(eval_points)
    for i, eval_point in enumerate(eval_points):
        if torch.any(eval_point == points): # Check if eval_point is in points
            eval[i] = values[torch.where(eval_point == points)[0][0]]
        else:
            inv_diff = 1/((domain[1]-eval_point) - points)
            val = inv_diff * weights
            eval[i] = torch.sum(val * values) / torch.sum(val)
    return eval 

def cheb_2d_impl(eval_points: torch.Tensor, chebyshev_size: tuple, chebyshev_values: torch.Tensor, domain: tuple):
    '''
    Chebyshev interpolation in 2D. (use sample_chebyshev_points_3 to sample) \n
    :param Tensor eval_points: (n*m) x 2 (n x_nodes * m y_nodes)
    :param Tensor values: Values at chebyshev points: (n*m) Tensor (n x_nodes, m y_nodes)
    :param tuple domain: (x_min, x_max, y_min, y_max)
    '''
    x_nodes, y_nodes = chebyshev_size
    assert len(chebyshev_values) == x_nodes * y_nodes, f"chebyshev_values ({len(chebyshev_values)}) does not match expected size {x_nodes * y_nodes} for chebyshev_size {chebyshev_size}."
    
    assert eval_points.device == chebyshev_values.device, f"eval_points ({eval_points.device}) and values ({chebyshev_values.device}) are not on the same device."
    print("Chebyshev 2D Interpolation implentation doesn't perform well on the domain boundary. Investigation is required.")

    eval_x = eval_points[:, 0]
    eval_y = eval_points[:, 1]
    res1 = torch.zeros((y_nodes, len(eval_points)), device=eval_points.device)
    for i in range(y_nodes):
        res1[i] = cheb_1d_impl(eval_x, chebyshev_values[i*x_nodes:(i+1)*x_nodes], domain[0:2])
    res2 = torch.zeros(len(eval_points), device=eval_points.device)
    for i in range(len(eval_points)):
        res2[i] = cheb_1d_impl(eval_y[i:i+1], res1[:, i], domain[2:4])
    return res2

def cheb_2d_plot_debugger(eval_points: torch.Tensor, cheb_itpl_values: torch.Tensor, gt_values: torch.Tensor = None):
    '''
    Debugger for chebyshev 2D interpolation.

    Parameters:
        eval_points (torch.Tensor): Points at which values where interpolated.
        cheb_itpl_values (torch.Tensor): Interpolated Chebyshev values.
        gt_values (torch.Tensor): Optional paramter for ground truth values at evaluated points.
    '''

    plot_points(points=eval_points, values=cheb_itpl_values, cmap="viridis", title="Interpolated Chebyshev values")
    if gt_values is not None:
        plot_points(points=eval_points, values=gt_values, cmap="viridis", title="Ground Truth values")


def cheb_2d_impl_convergence_rate_debugger(eval_points: torch.Tensor, evaluation_values: torch.Tensor, chebyshev_values: list[torch.Tensor], chebyshev_sizes: list[tuple], domain: tuple):
    error_rates = torch.zeros(len(chebyshev_sizes))
    h_values = list(map(lambda f: f[0], chebyshev_sizes))
    for i, chebyshev_size in enumerate(chebyshev_sizes):
        approx_eval = cheb_2d_impl(eval_points=eval_points, chebyshev_values=chebyshev_values[i], chebyshev_size=chebyshev_size, domain=domain)
        error_rates[i] = torch.sum(torch.abs(approx_eval-evaluation_values))
    
    log_h = np.log(h_values)
    log_E = np.log(error_rates)
    slope, intercept, _, _, _ = linregress(log_h, log_E)
    p = slope

    h = np.linspace(3, 100, 100)
    errors = h**slope * np.exp(intercept)
    plot_convergence_rate(h=h, error=errors, discrete_h_values=h_values, discrete_error_values=error_rates, p=p)


def clenshaw_curtis_weights(n):
    '''
    n: we define n + 1 quadrature nodes, and calculate weights using k = 0, 1, ..., n
    '''
    c = torch.ones(n+1)
    c[1:-1] = 2
    
    n_2 = math.floor(n/2)
    b = torch.ones(n_2)
    b[0:-1] = 2

    k = torch.linspace(0,n,n+1) 
    j = torch.linspace(1, n_2, n_2)

    kk, jj = torch.meshgrid(k, j, indexing='ij')
    cos_mat = torch.cos(2*kk*jj*math.pi/n)
    norm = (1/(4*jj**2-1))
    weights = 1 - torch.matmul(cos_mat * norm, b)
    weights *= c/n
    return weights 

def clenshaw_curtis_weights_2d(num_points):
    '''
    Designed to be coupled with sample_chebyshev_points_3
    '''
    x_num, y_num = num_points
    w_x = clenshaw_curtis_weights(x_num)
    w_y = clenshaw_curtis_weights(y_num)
    # w_y = torch.ones_like(w_y) 
    weights = w_y[:, None]*w_x # Returns in ij format, i.e. weights[i, j] gets you weights at x_j, y_i (how a matrix parallels the mesh geometrically)
    return weights.ravel()   #Unravels it so that weights[start:start+x_num] gives you the x_num weights for a specific y value scaled by that y values own weight.


if __name__ == "__main__":


    #  Example usage
    f = lambda x, y: x+y
    a, b = 0, 1.0
    c, d = 0, 1.0
    n = 20
    
    area_ratio = (b-a)*(d-c)/(4)

    weights = clenshaw_curtis_weights_2d((n-1,n-1)) * area_ratio
    points= sample_points((a, b, c, d), (n, n), "chebyshev")
    print("Chebyshev points: ",points)
    f_vals = f(points[:,0], points[:,1])
    eval = (f_vals*weights).sum()
    gt = (torch.exp(torch.tensor(b))-torch.exp(torch.tensor(a)))**2
    print(eval, gt)
    print(eval-gt)

    eval_points= sample_points((a, b, c, d), (2*n, 2*n), "chebyshev")
    eval_points = sample_points((a, b, c, d), (33, 33), "uniform")
    values = cheb_2d_impl(eval_points=eval_points, chebyshev_values=f_vals, chebyshev_size=(n, n), domain=(a, b, c, d))
    plot_points(points, f_vals, title="Chebyshev Interpolation 2D")
    plot_points(eval_points, values, title="Chebyshev Interpolation 2D")


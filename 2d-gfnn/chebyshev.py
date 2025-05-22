import math
import torch

from data_generation_utils import sample_random_mesh_points
from generate_data import source_term
from plot_utils import plot_points

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
    # xx, yy = torch.meshgrid(points_x, points_y, indexing='ij')
    xy = torch.zeros((x_num, y_num, 2))
    for i in range(x_num):
        for j in range(y_num):
            xy[i,j] = torch.tensor([points_x[i], points_y[j]])
    # result = torch.column_stack((xx.ravel(), yy.ravel()))
    return xy

def cheb_1d_points(domain, num_points: int):
    x_min, x_max = domain
    points_x = torch.linspace(0, num_points-1, num_points) * torch.pi / (num_points-1)
    points_x = torch.cos(points_x)
    points_x += 1
    points_x /= 2
    points_x = points_x * (x_max-x_min) + x_min
    return points_x


def cheb_weights(n):
    '''
    Calculates the Chebyshev weights for n points of the second kind. n is the number of nodes, not the final index.

    Parameters 
    ----------------
    n: number of nodes
    '''
    weights = torch.linspace(0, n-1, n)
    weights = torch.pow(-1, weights)
    weights[0] = weights[-1] = 0.5
    return weights

def cheb_1d_impl(eval_points, values, domain):
    n = len(values)
    points = cheb_1d_points(domain, n)
    weights = cheb_weights(n)
    eval = torch.zeros_like(eval_points)

    for i, eval_point in enumerate(eval_points):
        inv_diff = 1/(eval_point - points)
        val = inv_diff * weights
        eval[i] = torch.sum(val * values) / torch.sum(val)

    # plot_points(torch.vstack((torch.zeros_like(points), points)).T, values=values)
    # plot_points(torch.vstack((torch.zeros_like(eval_points), eval_points)).T, values= eval)
    return eval 

def cheb_2d_impl(eval_points, values, domain):
    '''
    values: n x m (n x_nodes, m y_nodes)
    '''
    eval_x = eval_points[:, 0]
    eval_y = eval_points[:, 1]
    x_nodes = len(values)
    
    #for each y evaluate x
    res1 = torch.zeros((x_nodes, len(eval_points)))
    for i in range(x_nodes):
        res1[i] = cheb_1d_impl(eval_y, values[i, :], domain[2:4])

    res2 = torch.zeros(len(eval_points))
    for i in range(len(eval_points)):
        res2[i] = cheb_1d_impl(eval_x[i:i+1], res1[:, i], domain[0:2])
    return res2

if __name__ == "__main__":
    domain = (0,1,0,1)
    eval_points = sample_random_mesh_points(domain, 30)
    cheb_points = sample_chebyshev_points_2(domain, (20, 20))
    
    values = source_term(cheb_points)
    eval_values = cheb_2d_impl(eval_points=eval_points, values=values, domain=domain)
    plot_points(cheb_points.view(400, 2), values.view(400))
    plot_points(eval_points, eval_values)

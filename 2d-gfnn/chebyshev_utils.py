import math
import torch

from data_generation_utils import sample_chebyshev_points, sample_chebyshev_points_3, sample_random_mesh_points
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
    xy = torch.zeros((x_num, y_num, 2))
    for i in range(x_num):
        for j in range(y_num):
            xy[i,j] = torch.tensor([points_x[i], points_y[j]])
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
    f = lambda x, y: torch.exp(x + y)
    a, b = -1000, 1.0
    c, d = -1000, 1.0
    n = 4000
    
    area_ratio = (b-a)*(d-c)/(4)

    weights = clenshaw_curtis_weights_2d((n-1,n-1)) * area_ratio
    points= sample_chebyshev_points_3((a, b, c, d), (n, n))
    f_vals = f(points[:,0], points[:,1])
    eval = (f_vals*weights).sum()
    gt = (torch.exp(torch.tensor(b))-torch.exp(torch.tensor(a)))**2
    print(eval, gt)
    print(eval-gt)


    assert False

    domain = (0,1,0,1)
    eval_points = sample_random_mesh_points(domain, 30)
    cheb_points = sample_chebyshev_points_2(domain, (20, 20))
    
    def source_term(points):
        x, y = points[..., 0], points[..., 1]
        print(x)
        return x

    values = source_term(cheb_points)
    eval_values = cheb_2d_impl(eval_points=eval_points, values=values, domain=domain)
    plot_points(cheb_points.view(400, 2), values.view(400))
    plot_points(eval_points, eval_values)

import math
import os
import typing
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def sample_random_mesh_points(domain, num_points, boundary: bool = False):
    x_min, x_max, y_min, y_max = domain
    if boundary:
        half = num_points//2
        x = torch.rand(half) * (x_max-x_min) + x_min
        y = torch.rand(half) * (y_max-y_min) + y_min
        results = []
        for i, b in enumerate(domain):
            j = i % 2
            ind = half//2*j
            b_tensor = torch.full((half//2,), b)
            paired = torch.column_stack((b_tensor, y[ind:(ind+half//2)])) if i < 2 else torch.column_stack((x[ind:(ind+half//2)], b_tensor))
            results.append(paired)
        return torch.cat(results, dim=0)
    else:
        x = torch.rand(num_points) * (x_max-x_min) + x_min
        y = torch.rand(num_points) * (y_max-y_min) + y_min
        return torch.vstack((x, y)).T


def sample_uniform_mesh_points(domain, num_points: tuple, boundary: bool = False):
    x_min, x_max, y_min, y_max = domain
    x_num_points, y_num_points = num_points
    x = torch.linspace(x_min, x_max, x_num_points)
    y = torch.linspace(y_min, y_max, y_num_points)
    if boundary:
        results = []
        for i, b in enumerate(domain):
            b_tensor = torch.full_like(x, b)
            paired = torch.column_stack((b_tensor, y)) if i < 2 else torch.column_stack((x, b_tensor))
            results.append(paired)
        return torch.cat(results, dim=0)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    result = torch.column_stack((xx.ravel(), yy.ravel()))
    return result


def sample_chebyshev_points(domain, num_points: tuple):
    '''
    Samples Chebyshev points and returns output with shape (num_points x 2).
    To sample Chebyshev points with output with shape (num_points x num_points), see sample_chebyshev_points_2.
    '''
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
    xx, yy = torch.meshgrid(points_x, points_y, indexing='ij')
    result = torch.column_stack((xx.ravel(), yy.ravel()))
    print(xx, yy)
    assert False
    return result

def sample_chebyshev_points_3(domain, num_points: tuple):
    '''
    Samples Chebyshev points and returns output points with shape (num_points x 2).
    Differs from sample_chebyshev_points: this produces an ordered list of points organized by (x_low,y_low) to (x_high, y_high), where 
        points[start:start+x_num] gives us all the ordered x_values paired with a single y_value. The points are ordered accordingly from y_low to y_high.
    To sample Chebyshev points with output with shape (num_points x num_points), see sample_chebyshev_points_2.
    '''
    x_num, y_num = num_points
    x_min, x_max, y_min, y_max = domain
    points_x = torch.linspace(x_num-1, 0, x_num) * torch.pi / (x_num-1)
    points_x = torch.cos(points_x)
    points_y = torch.linspace(y_num-1, 0, y_num) * torch.pi / (y_num-1)
    points_y = torch.cos(points_y)
    points_x += 1
    points_x /= 2
    points_y += 1
    points_y /= 2
    points_x = points_x * (x_max-x_min) + x_min
    points_y = points_y * (y_max-y_min) + y_min
    xx, yy = torch.meshgrid(points_x, points_y, indexing='xy')
    result = torch.column_stack((xx.ravel(), yy.ravel()))
    return result

def generate_points(domain, u_gt_func, f_func, save_dir: str, num_points: tuple, f_qudrature_num_points: tuple, 
                    training_data: bool= True, u_chebyshev_mesh: bool = True, uniform_f_quadrature: bool = False,  
                    u_gt_expr: str = None, f_func_expr: str = None, u_bnd_expr: str = None):
    
    if not os.path.exists(save_dir):
        raise Exception("The directory " + save_dir + " doesn't exist.")
    print("Generating points to save into dir: " + save_dir)
    # Sample mesh points
    if u_chebyshev_mesh:
        # Sample mesh points as chebyshev points
        x_num_nodes, y_num_nodes = num_points
        mesh_points = sample_chebyshev_points_3(domain, num_points=(x_num_nodes, y_num_nodes))
    else:
        # Sample random points
        assert len(num_points) == 1 
        mesh_points = sample_random_mesh_points(domain, num_points[0])

    #Get u_values from mesh points
    u_values = u_gt_func(mesh_points)

    if uniform_f_quadrature:
        #Get f_values from uniform mesh points
        x_num_nodes, y_num_nodes = f_qudrature_num_points
        f_mesh = sample_uniform_mesh_points(domain, (x_num_nodes, y_num_nodes))
        f_values = f_func(f_mesh)
    else:
        #Get f_values from previously defined mesh points
        x_num_nodes, y_num_nodes = f_qudrature_num_points
        f_mesh_points = sample_chebyshev_points_3(domain, num_points=(x_num_nodes, y_num_nodes))
        f_mesh = f_mesh_points
        f_values = f_func(f_mesh_points)
    
    data = {'coordinates': mesh_points, 'u_values': u_values, 'f_values': f_values, 
            "f_mesh": f_mesh, "uniform_quadrature": uniform_f_quadrature, "chebyshev_bool": u_chebyshev_mesh, 
            "u_gt_func_expr": u_gt_expr, "f_func_str_expr": f_func_expr, "u_bnd_func_expr": u_bnd_expr, "domain": domain}
    
    file_suffix = "_train.pt" if training_data else "_test.pt"
    torch.save(data, save_dir + "data"+file_suffix)
    print("Saved generated points into " + save_dir + ".")
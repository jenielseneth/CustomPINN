import math
import os
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

# def sample_mesh_points(domain, num_points, uniform: bool = False, boundary: bool = False):
#     x_min, x_max, y_min, y_max = domain
#     if boundary:
#         half = num_points//2
#         x = torch.rand(half) * (x_max-x_min) + x_min
#         y = torch.rand(half) * (y_max-y_min) + y_min
#         results = []
#         for i, b in enumerate(domain):
#             j = i % 2
#             ind = half//2*j
#             b_tensor = torch.full((half//2,), b)
#             paired = torch.column_stack((b_tensor, y[ind:(ind+half//2)])) if i < 2 else torch.column_stack((x[ind:(ind+half//2)], b_tensor))
#             results.append(paired)
#         return torch.cat(results, dim=0)
#     if not uniform:
#         x = torch.rand(num_points) * (x_max-x_min) + x_min
#         y = torch.rand(num_points) * (y_max-y_min) + y_min
#         return torch.vstack((x, y)).T
#     else:
#         spacing = math.floor(math.sqrt(num_points))
#         x = torch.linspace(x_min, x_max, spacing)
#         y = torch.linspace(y_min, y_max, spacing)
#         if boundary:
#             results = []
#             for i, b in enumerate(domain):
#                 b_tensor = torch.full_like(x, b)
#                 paired = torch.column_stack((b_tensor, y)) if i < 2 else torch.column_stack((x, b_tensor))
#                 results.append(paired)
#             return torch.cat(results, dim=0)
#         xx, yy = torch.meshgrid(x, y, indexing='ij')
#         result = torch.column_stack((xx.ravel(), yy.ravel()))
#         return result

def separate_collocation_boundary_points(domain, points):
    boundary_ind = []
    collocation_ind = []
    for i, point in enumerate(points):
        if point[0] in domain[0:2] or point[1] in domain[2:4]:
            boundary_ind.append(i)
        else:
            collocation_ind.append(i)
    boundary_data = points[boundary_ind]
    collocation_data = points[collocation_ind]
    return collocation_data, boundary_data

def generate_points(domain, eval_u_func, dir: str, num_col_points: int, num_bnd_points: int = 20, chebyshev:bool = False, training_data: bool= True, boundary_value: float = None):
    if not os.path.exists(dir):
        raise Exception("The directory " + dir + " doesn't exist.")
    x_min, x_max, y_min, y_max = domain
    print("Generating points to save into dir: " + dir)
    # Sample mesh points
    if chebyshev:
        num_points_sqrt = math.floor(math.sqrt(num_col_points))
        print("When calculating Chebyshev points, the number of boundary points is automatically included in the calculation. num-bnd_points is therefore void, and is automatically repplaced with the number: " + f'{num_points_sqrt*4-4}' + ".")
        mesh_points = sample_chebyshev_points(domain, num_points=(num_points_sqrt, num_points_sqrt))
        collocation_points,boundary_points = separate_collocation_boundary_points(domain, mesh_points)
    else:
        ep = 1e-5
        collocation_domain = (x_min+ep, x_max-ep, y_min+ep, y_max)
        collocation_points = sample_random_mesh_points(collocation_domain, num_col_points)
        boundary_points = sample_random_mesh_points(domain, num_bnd_points, boundary=True)

    collocation_u_values = []
    boundary_u_values = []
    
    collocation_u_values = eval_u_func(collocation_points)
    boundary_u_values = eval_u_func(boundary_points)
    collocation_points = {'coordinates': collocation_points, 'values': collocation_u_values}
    boundary_points = {'coordinates': boundary_points, 'values': boundary_u_values}
    file_suffix = "_train.pt" if training_data else "_test.pt"
    torch.save(collocation_points, dir + "collocation"+file_suffix)
    torch.save(boundary_points, dir + "boundary"+file_suffix)
    print("Saved generated points into " + dir + ".")


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
    return result


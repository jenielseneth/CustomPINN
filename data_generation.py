import math
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def sample_mesh_points(x_min, x_max, y_min, y_max, num_points, uniform: bool = False):
    if not uniform:
        x = torch.rand(num_points, requires_grad=True) * (x_max-x_min) + x_min
        y = torch.rand(num_points, requires_grad=True) * (y_max-y_min) + y_min
        # x = np.random.uniform(x_min, x_max, num_points)
        # y = np.random.uniform(y_min, y_max, num_points)
        return torch.vstack((x, y)).T
    else:
        spacing = math.floor(math.sqrt(num_points))
        x = torch.linspace(x_min, x_max, spacing, requires_grad=True)
        y = torch.linspace(y_min, y_max, spacing, requires_grad=True)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        result = torch.column_stack((xx.ravel(), yy.ravel()))
        return result
    
def generate_points(domain, num_points, evaluate_u, chebyshev:bool, file_name: str, boundary_value: float):
    x_min, x_max, y_min, y_max = domain
    print("Generating points to save into file: " + file_name)
    # Sample mesh points
    if chebyshev:
        num_points_sqrt = math.floor(math.sqrt(num_points))
        mesh_points = sample_chebyshev_points(domain, num_points=(num_points_sqrt, num_points_sqrt))
    else:
        mesh_points = sample_mesh_points(x_min, x_max, y_min, y_max, num_points)
    
    # Evaluate the solution u at the sampled points
    points = torch.zeros((num_points, 2))
    u_values = torch.zeros(mesh_points.shape[0])
    data = {'coordinates': points, 'values': u_values}
    for i in tqdm(range(mesh_points.shape[0])):

        if mesh_points[i, 0] in (x_min, x_max) or mesh_points[i, 1] in (y_min, y_max):
            print(mesh_points)
            u = boundary_value
        else:
            u = evaluate_u(mesh_points[i], domain)
        data["coordinates"][i,0] = mesh_points[i, 0]
        data["coordinates"][i,1] = mesh_points[i, 1]
        data["values"][i] = u
    
    torch.save(data, file_name)
    print("Saved generated points into " + file_name)


def sample_chebyshev_points(domain, num_points: tuple):
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


if __name__ == "__main__":
    generate_points()
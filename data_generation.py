import torch
import numpy as np
from tqdm import tqdm


def sample_mesh_points(x_min, x_max, y_min, y_max, num_points, uniform: bool = False):
    if not uniform:
        x = torch.rand(num_points) * (x_max-x_min) + x_min
        y = torch.rand(num_points) * (y_max-y_min) + y_min
        # x = np.random.uniform(x_min, x_max, num_points)
        # y = np.random.uniform(y_min, y_max, num_points)
        return torch.vstack((x, y)).T
    else:
        spacing = np.floor(np.sqrt(num_points)).astype('int32')
        x = torch.linspace(x_min, x_max, spacing)
        y = torch.linspace(y_min, y_max, spacing)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        result = torch.column_stack((xx.ravel(), yy.ravel()))
        return result
    
def generate_points(x_min, x_max, y_min, y_max, num_points, evaluate_u, file_name: str):
    print("Generating points to save into file: " + file_name)
    # Sample mesh points
    mesh_points = sample_mesh_points(x_min, x_max, y_min, y_max, num_points)
    
    # Define the domain for integration
    domain = (x_min, x_max, y_min, y_max)
    
    # Evaluate the solution u at the sampled points
    points = torch.zeros((num_points, 2))
    u_values = torch.zeros(mesh_points.shape[0])
    data = {'coordinates': points, 'values': u_values}
    for i in tqdm(range(mesh_points.shape[0])):

        u = evaluate_u(mesh_points[i], domain)
        data["coordinates"][i,0] = mesh_points[i, 0]
        data["coordinates"][i,1] = mesh_points[i, 1]
        print(u)
        data["values"][i] = u
    
    torch.save(data, file_name)
    print("Saved generated points into " + file_name)


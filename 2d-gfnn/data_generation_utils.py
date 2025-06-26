import math
import os
import typing
import torch
import numpy as np
from tqdm import tqdm
from expr_generation_utils import expr_to_func, func_input_wrapper
from constants_utils import DataGenerationParameters
from random_utils import log_dict_as_json
import matplotlib.pyplot as plt
import sympy

def gcd_chebyshev_mesh_size(chebyshev_mesh_size: tuple):
    '''
    This is the default function for generating a corresponding mesh size who has a gcd of 1 corresponding to the input chebyshev_mesh_size.
    The purpose is to avoid overlapping chebyshev nodes of the second kind anywhere except the corner points.

    :param tuple chebyshev_mesh_size: 2 size tuple specifying the x- and y-axis sizes of your mesh.
    :return output_mesh_size: 2 size tuple.
    '''
    output_mesh_size = tuple(map(lambda x: x - 1, chebyshev_mesh_size))
    return output_mesh_size

def _sample_random_mesh_points(domain, num_points, boundary: bool = False):
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


def _sample_uniform_mesh_points(domain, num_points: tuple, boundary: bool = False):
    '''
    Samples uniform mesh points and returns Tensor with shape (x_num_points * y_num_points x 2).
    '''
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


def _sample_chebyshev_points(domain, num_points: tuple, boundary: bool = False):
    '''
    Samples Chebyshev points and returns output points with shape (num_points x 2).
    This function produces an ordered list of points organized by (x_low,y_low) to (x_high, y_high), where 
        points[start:start+x_num] gives us all the ordered x_values paired with a single y_value. The 
        points are ordered accordingly from y_low to y_high.
    To sample Chebyshev points with output with shape (num_points x num_points), see sample_chebyshev_points_2.
    '''
    x_num, y_num = num_points
    x_min, x_max, y_min, y_max = domain
    #Reverse torch.linspace to ensure points_x and points_y have order from low to high.
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
    
    if boundary:
        bnd_x = torch.tensor([x_min, x_max], dtype=torch.float32)
        bnd_y = torch.tensor([y_min, y_max], dtype=torch.float32)

        xx, yy = torch.meshgrid(points_x, bnd_y, indexing='xy')
        result = torch.column_stack((xx.ravel(), yy.ravel()))
        yy, xx = torch.meshgrid(points_y, bnd_x, indexing='xy')
        result = torch.vstack((result, torch.column_stack((xx.ravel(), yy.ravel()))))
    else:
        xx, yy = torch.meshgrid(points_x, points_y, indexing='xy')
        result = torch.column_stack((xx.ravel(), yy.ravel()))

    return result

def sample_points(domain, mesh_size: tuple, mesh_type: typing.Literal["chebyshev", "uniform", "random"] = "chebyshev", boundary: bool = False):
    """
    Samples points in a given domain using specified mesh type.
    
    Args:
        domain (tuple): The spatial domain defined as (x_min, x_max, y_min, y_max).
        num_points (tuple): Number of points to sample in each dimension (x_num, y_num).
        mesh_type (str): Type of mesh to use for sampling. Options are "chebyshev", "uniform", or "random".
        boundary (bool): If True, samples points only on the boundary of the domain.
    
    Returns:
        torch.Tensor: Sampled points in the domain.
    """
    if mesh_type == "chebyshev":
        mesh = _sample_chebyshev_points(domain, mesh_size, boundary)
    elif mesh_type == "uniform":
        mesh = _sample_uniform_mesh_points(domain, mesh_size, boundary)
    elif mesh_type == "random":
        mesh = _sample_random_mesh_points(domain, mesh_size[0], boundary)
    else:
        raise ValueError("Invalid mesh_type. Choose from 'chebyshev', 'uniform', or 'random'.")
    return mesh


def generate_points(domain, 
                    u_exprs: list[sympy.Expr], f_exprs: list[sympy.Expr],
                    save_dir: str, u_mesh_size: tuple, f_mesh_size: tuple, 
                    training_data: bool= True, 
                    u_mesh_type: typing.Literal["chebyshev", "uniform", "random"] = "chebyshev",
                    f_mesh_type: typing.Literal["chebyshev", "uniform", "random"] = "chebyshev",
                    a_expression: sympy.Expr = None, u_bnd_expr: sympy.Expr = None):
    """
    Generates and saves mesh points, ground truth solution values, and right-hand side function values for a given domain,
    supporting both Chebyshev and uniform quadrature meshes. The generated data is saved as a PyTorch file for use in
    training or testing PINN (Physics-Informed Neural Network) models.
    Args:
        domain: The spatial domain over which to generate points (typically a tuple or list specifying bounds).
        u_gt_funcs (list of callables): List of ground truth solution functions u(x, y) to evaluate at mesh points.
        f_funcs (list of callables): List of right-hand side functions f(x, y) to evaluate at quadrature points.
        save_dir (str): Directory path where the generated data will be saved.
        num_points (tuple): Number of mesh points in each spatial dimension for the solution (e.g., (Nx, Ny)).
        f_qudrature_num_points (tuple): Number of quadrature points in each spatial dimension for f (e.g., (Nqx, Nqy)).
        training_data (bool, optional): If True, saves data as training set; otherwise, as test set. Default is True.
        u_chebyshev_mesh (bool, optional): If True, uses Chebyshev points for the solution mesh; otherwise, uses random points. Default is True.
        uniform_f_quadrature (bool, optional): If True, uses a uniform mesh for f quadrature; otherwise, uses Chebyshev points. Default is False.
        u_gt_exprs (list of str, optional): List of string expressions of the ground truth solution functions (for metadata/documentation).
        f_func_exprs (list of str, optional): List of string expressions of the right-hand side functions (for metadata/documentation).
        u_bnd_expr (str, optional): String expression of the boundary condition function (for metadata/documentation).
    """


    #Get from the expressions the appropriate functions.
    u_funcs = func_input_wrapper(expr_to_func(u_exprs))
    f_funcs = func_input_wrapper(expr_to_func(f_exprs))


    ind_data = []

    if not os.path.exists(save_dir):
        raise Exception("The directory " + save_dir + " doesn't exist.")
    print("Generating points to save into dir: " + save_dir)
    for i, (u_gt_func, f_func) in tqdm(enumerate(zip(u_funcs, f_funcs)), total=len(u_funcs)):
        # Sample mesh points
        if u_mesh_type == "chebyshev":
            assert len(u_mesh_size) == 2
        elif u_mesh_type == "uniform":
            assert len(u_mesh_size) == 2
        elif u_mesh_type == "random":
            assert len(u_mesh_size) == 1 
        else:
            raise ValueError("Invalid u_mesh_type. Choose from 'chebyshev', 'uniform', or 'random'.")
        
        mesh_points = sample_points(domain=domain, mesh_size=u_mesh_size, mesh_type=u_mesh_type, boundary=False)
        #Get u_values from mesh points
        u_values = u_gt_func(mesh_points)

        if f_mesh_type == "chebyshev":
            assert len(f_mesh_size) == 2
        elif f_mesh_type == "uniform":
            assert len(f_mesh_size) == 2
        elif f_mesh_type == "random":
            assert len(f_mesh_size) == 1 

        f_mesh = sample_points(domain=domain, mesh_size=f_mesh_size, mesh_type=f_mesh_type, boundary=False)
        f_values = f_func(f_mesh)

        #Return the interior and boundary indices of the mesh points.

        ind_data.append({'coordinates': mesh_points, 'u_values': u_values, 'f_values': f_values, 
            "f_mesh": f_mesh, "f_mesh_type": f_mesh_type, "u_mesh_type": u_mesh_type, 
            "u_gt_func_expr": str(u_exprs[i]), "f_func_str_expr": str(f_exprs[i]), "u_bnd_func_expr": str(u_bnd_expr), "domain": domain})

    #Concatenate all the data into a single dictionary.
    #Use vstack to concatenate mesh_points and f_mesh, as they are 2D tensors.
    mesh_points = torch.vstack([data['coordinates'] for data in ind_data]) # size: (N, 2)
    u_values = torch.hstack([data['u_values'] for data in ind_data]) # size: (N,)

    f_mesh = torch.cat([data['f_mesh'][None, ...] for data in ind_data]) # size: (num_expr, f_mesh_size, 2)
    f_values = torch.vstack([data['f_values'] for data in ind_data]) # size: (num_expr, f_mesh_size)

    u_exprs = [data['u_gt_func_expr'] for data in ind_data]
    f_exprs = [data['f_func_str_expr'] for data in ind_data]
    u_bnd_exprs = [data['u_bnd_func_expr'] for data in ind_data]

    start = 0
    data_addresses = []
    for data in ind_data:
        address = (start, start + len(data['coordinates']))
        start += len(data['coordinates'])
        data_addresses.append(address)


    data = {'coordinates': mesh_points, 'u_values': u_values, 'f_values': f_values, 
            "f_meshes": f_mesh, "f_mesh_type": f_mesh_type, "u_mesh_type": u_mesh_type, 
            "u_mesh_size": u_mesh_size, "f_mesh_size": f_mesh_size,
            "u_gt_func_exprs": u_exprs, "f_func_str_exprs": f_exprs, "u_bnd_func_exprs": u_bnd_exprs, 
            "data_addresses": data_addresses, "domain": domain}
    

    dg_params = DataGenerationParameters(domain=domain,
                                         evaluation_mesh_size=u_mesh_size,
                                         evaluation_mesh_type=u_mesh_type,
                                         integration_mesh_size=f_mesh_size,
                                         integration_mesh_type=f_mesh_type,
                                        u_func_exprs=u_exprs,
                                        f_func_exprs=f_exprs,
                                        u_bnd_expr=u_bnd_expr,
                                        a_diffusion_expr=a_expression,
                                        )
    
    info_file_name = "train_" + "params.json" if training_data else "test_" + "params.json"
    log_dict_as_json(dg_params.get_dict(), save_dir + info_file_name)
    
    file_suffix = "_train.pt" if training_data else "_test.pt"
    torch.save(data, save_dir + "data"+file_suffix)
    print("Saved generated points into " + save_dir + ".")
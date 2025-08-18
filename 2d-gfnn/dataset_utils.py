from dataclasses import dataclass, field
from typing import Iterable, TypeVar, Generic
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from plot_utils import plot_points
from constants_utils import mesh_type
import sympy
import logging
import random


logger = logging.getLogger(__name__)



def get_interior_boundary_idx(domain, mesh):
    '''
    Returns the indices of collocation and boundary points in the given points.

    :param tuple domain: (x_min, x_max, y_min, y_max) tuple.
    :param Tensor mesh: b x 2 Tensor.

    :return interior_ind: (intr) Tensor
    :return boundary_ind: ((b - intr)) Tensor
    '''
    boundary_idx = []
    interior_idx = []
    for i, point in enumerate(mesh):
        if point[0] in domain[0:2] or point[1] in domain[2:4]:
            boundary_idx.append(i)
        else:
            interior_idx.append(i)
    return interior_idx, boundary_idx

def get_interior_mesh(domain, mesh):
    '''
    Returns the interior of the given mesh mesh.

    :param tuple domain: (x_min, x_max, y_min, y_max) tuple.
    :param Tensor mesh: b x 2 Tensor.

    :return mesh: intr x 2 Tensor
    '''
    intr, _ = get_interior_boundary_idx(domain=domain, mesh=mesh)
    return mesh[intr]

def get_boundary_mesh(domain, mesh):
    '''
    Returns the boundary of the given mesh.

    :param tuple domain: (x_min, x_max, y_min, y_max) tuple.
    :param Tensor mesh: b x 2 Tensor.

    :return mesh: bnd x 2 Tensor
    '''
    _, bnd = get_interior_boundary_idx(domain=domain, mesh=mesh)
    return mesh[bnd]

def get_corners_idx(domain, mesh):
    '''
    Returns the indices of the non-corners and corners of a mesh of points.

    :param tuple domain: (x_min, x_max, y_min, y_max) tuple.
    :param Tensor mesh: b x 2 Tensor.

    :return non_corners_idx: (n_c) Tensor
    :return corners_idx: ((b - n_c)) Tensor
    '''
    x_min, x_max, y_min, y_max = domain
    corners = torch.tensor([[x_min, y_min], [x_min, y_max], [x_max, y_min], [x_max, y_max]])
    corner_mask = torch.any(torch.all(mesh[:, None] == corners, dim=-1), dim=1)
    corners_idx = (corner_mask).nonzero().flatten()
    non_corners_idx = (~corner_mask).nonzero().flatten()
    return non_corners_idx, corners_idx

def get_non_corners_mesh(domain, mesh):
    '''
    Returns the mesh with corners points removed.

    :param tuple domain: (x_min, x_max, y_min, y_max) tuple.
    :param Tensor mesh: n_c x 2 Tensor.

    :return mesh: 
    '''
    n_c, _ = get_corners_idx(domain=domain, mesh=mesh)
    return mesh[n_c]


@dataclass
class GreensConstantsDataclass:
    '''
    Dataclass containing constant data values to be accessed.
    '''
    domain: tuple
    evaluation_mesh_type: mesh_type
    integration_mesh_type: mesh_type
    evaluation_mesh_sizes: tuple
    integration_mesh_sizes: tuple

    def to_device(self, device):
        return


@dataclass
class DatasetReturnClass:
    crd: torch.Tensor
    u_vals: torch.Tensor 
    f_mesh_idx: torch.Tensor
    f_vals_idx: torch.Tensor 
class GreenPINNDataset(Dataset):
    '''
    Wrapper to retrieve all datasets and store them into one main dataset wrapper.
    All datasets are concatenated with each other.
    We use a pointer system for mapping u_values with their corresponding f_meshes and f_values to avoid duplicate copies of the data.

    Attributes:
        self.constants (GreensConstantsDataclass): Contains the constants of the dataset.
        self.length (int): Length of the dataset.
        self.data_length (int): Total amount of source terms.
        self.evaluation_mesh (Tensor): b x 2 Tensor of evaluation points.
        self.evaluation_values (Tensor): b x 1 Tensor of evaluation values.
        self.f_meshes (Tensor): f_size x 2 Tensor of source term meshes.
        self.f_values (Tensor): f_size x 1 Tensor of source term values.
        self.f_inds (list[int]): List of indices to map evaluation points to their corresponding source term mesh and values.
        self.sub_lengths (list[tuple]): List of tuples containing the start and end indices of each subdataset for a given source term f.
    
    '''
    def __init__(self, data_file_path, data_file_name: str, subset_idx: int = None):
        data = torch.load(data_file_path + data_file_name)

        self.constants = GreensConstantsDataclass(
            domain=data["domain"],
            evaluation_mesh_type=data["u_mesh_type"],
            integration_mesh_type=data["f_mesh_type"],
            evaluation_mesh_sizes=data["u_mesh_sizes"],
            integration_mesh_sizes=data["f_mesh_sizes"]
        )
        
        self.evaluation_mesh = data["coordinates"] # n_expr * (a_1 * u_mesh_size_1 + a_2 * u_mesh_size_2 + ... + a_m * u_mesh_size_m) x 2 Tensor of evaluation meshes.
        self.evaluation_values = data["u_values"] # n_expr * (a_1 * u_mesh_size_1 + a_2 * u_mesh_size_2 + ... + a_m * u_mesh_size_m) Tensor of ground truth evaluation values.
        self.f_meshes = data["f_meshes"] # n_expr * (f_mesh_size_1 + f_mesh_size_2 + ... + f_mesh_size_n  ) x 2 Tensor of integration meshes.
        self.f_values = data["f_values"] # n_expr * (f_mesh_size_1 + f_mesh_size_2 + ... + f_mesh_size_n  ) Tensor of source term values.
        self.u_length = len(self.evaluation_values) 
        self.f_length = len(self.f_values) 
        self.u_data_addresses = data["u_data_addresses"] # List of length n_expr (number of source terms) containing tuples (start, end) for each source term.
        self.f_data_addresses = data["f_data_addresses"] # List of length n_expr (number of source terms) containing tuples (start, end) for each source term.
        self.u_to_f_mesh_idx = data["u_to_f_mesh_idx"] # List of length u_length (number of source terms) containing the index of the f_mesh for each u coordinate point.
        self.u_point_to_expr_idx = data["u_point_to_expr_idx"] # List of length u_length (number of source terms) containing the index of the source term for each u coordinate point.
        self.mesh_addresses = data["mesh_size_addresses"] # List of tuples (start, end) for the evaluation points for each corresponding mesh size.
        
        # Calculate starting indices in relation to amount of total source terms for each new size f - i.e. calculating n_expr_1 + n_expr_2 + ... + n_expr_n   
        self.num_f_terms = []
        start = 0
        for n in data["num_f_terms"]:
            self.num_f_terms.append(start)
            start += n

        # u_inds (u_crd_i) -> source_term_index
        # self.u_inds = [0] * len(self.evaluation_mesh) # Map evaluation points to their corresponding source term index.
        # for i, address in enumerate(self.u_data_addresses):
        #     self.u_inds[slice(*address)] = [i] * (address[1]-address[0])
        
        if subset_idx is not None:
            assert subset_idx < self.u_length, f"sub_idx {subset_idx} is out of bounds for the dataset with length {self.u_length}."
            idxs = random.sample(range(0, self.u_length), subset_idx)
            self.evaluation_mesh = self.evaluation_mesh[idxs]
            self.evaluation_values = self.evaluation_values[idxs]
            self.u_length = subset_idx        
            self.u_to_f_mesh_idx = [data["u_to_f_mesh_idx"][i] for i in idxs]
            self.u_point_to_expr_idx = [data["u_point_to_expr_idx"][i] for i in idxs]



    def _str_to_sympy_expr(self, s: str):
        '''
        Converts strings to sympy expressions.
        '''
        expr = sympy.sympify(s)
        return expr

    def interior_points_dataset(self):
        '''
        Modify the dataset to only use interior points in the evaluation mesh, exclude the boundary points.
        '''
        intr, _ = get_interior_boundary_idx(domain=self.constants.domain, mesh=self.evaluation_mesh)
        self.evaluation_mesh = self.evaluation_mesh[intr]
        self.evaluation_values = self.evaluation_values[intr]
        self.u_length = len(self.evaluation_mesh)
        self.f_inds = [self.f_inds[i] for i in intr]

        ##Temporary solution to fix the sub_lengths for interior points.
        sub_length = self.u_length // len(self.u_addresses)
        for i in range(len(self.u_addresses)):
            self.u_addresses[i] = (i*sub_length, (i+1)*sub_length)

    def exclude_corners_dataset(self):
        '''
        Modify the dataset to exclude corner points in the evaluation mesh. 
        '''
        non_corners_idx, _ = get_corners_idx(domain=self.constants.domain, mesh=self.evaluation_mesh)
        self.evaluation_mesh = self.evaluation_mesh[non_corners_idx]
        self.evaluation_values = self.evaluation_values[non_corners_idx]
        self.u_length = len(self.evaluation_mesh)
        self.f_inds = [self.f_inds[i] for i in non_corners_idx]

        ##Temporary solution to fix the sub_lengths for interior points.
        sub_length = self.u_length // len(self.u_addresses)
        for i in range(len(self.u_addresses)):
            self.u_addresses[i] = (i*sub_length, (i+1)*sub_length)

    def plot_evaluation_mesh(self, idx):
        '''
        Debug tool to plot evaluation mesh.

        Parameters:
            idx: index to splice and plot subdataset.
        '''
        plot_points(points=self.evaluation_mesh[slice(*self.u_addresses[idx])])

    def plot_integration_mesh(self):
        '''
        Debug tool to plot evaluation mesh.
        '''
        plot_points(points=self.constants.integration_mesh)

    def get_f_mesh(self, u_idx: int | list[int]) -> list[torch.Tensor]:
        '''
        Returns the f_mesh for the given index of a u coordinate point.
        '''
        idx = self.u_to_f_mesh_idx[u_idx] # Get the index of the corresponding f_mesh for the given u coordinate point.
        if type(idx) == list:
            raise NotImplementedError("Not implemented yet for lists.")
            f_meshes = [self.f_meshes[i] for i in idx]
        else:
            f_meshes = self.f_meshes[idx]
        
        return f_meshes # (len(u_idx) x f_mesh_size x 2) Tensor of the source term mesh.
    
    def get_f_values(self, u_idx: int | list[int]) -> list[torch.Tensor]:
        '''
        Returns the f_values for the given index of a u coordinate point.
        '''
        mesh_idx = self.u_to_f_mesh_idx[u_idx] # Get the index of the source term for the given u coordinate point.
        value_idx = self.u_point_to_expr_idx[u_idx] # Get the index of the source term for the given u coordinate point.

        if type(u_idx) == list:
            raise NotImplementedError("Not implemented yet for lists.")
            f_values = [self.f_values[idx][value_idx[i]] for i, idx in enumerate(mesh_idx)]
        else:
            f_values = self.f_values[mesh_idx][value_idx]
        return f_values # (len(u_idx) x f_mesh_size) Tensor of the source term mesh.

    def __len__(self):
        # return total dataset size
        return self.u_length

    def __getitem__(self, index) -> DatasetReturnClass:
        '''
        Returns
        '''
        ret_item = DatasetReturnClass(
            crd=self.evaluation_mesh[index],
            u_vals=self.evaluation_values[index],
            f_mesh_idx=self.u_to_f_mesh_idx[index],
            f_vals_idx=self.u_point_to_expr_idx[index]
        )
        return ret_item

def greens_pinn_dataset_collate_fn(batch: list[DatasetReturnClass]) -> DatasetReturnClass:
    '''
    Collate function for the GreenPINNDataset.
    '''
    crd = torch.stack([item.crd for item in batch])
    u_vals = torch.stack([item.u_vals for item in batch])
    f_mesh_idx = [item.f_mesh_idx for item in batch]
    f_vals_idx = [item.f_vals_idx for item in batch]
    
    return DatasetReturnClass(crd=crd, u_vals=u_vals, f_mesh_idx=f_mesh_idx, f_vals_idx=f_vals_idx)

from dataclasses import dataclass, field
from typing import Iterable, TypeVar, Generic
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler
from tqdm import tqdm
from plot_utils import plot_points
from constants_utils import mesh_type
import sympy
import logging
from collections import Counter
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
    '''
    Returns:
    crd: Tensor of evaluation points.
    u_vals: Tensor of evaluation values.
    f_mesh_idx: int of the source term mesh.
    f_vals: Tensor of source term values.'''
    crd: torch.Tensor
    u_vals: torch.Tensor 
    f_mesh_idx: int | list[int]
    f_vals: torch.Tensor 
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
        self.evaluation_values = data["u_values"] #  n_expr * (a_1 * u_mesh_size_1 + a_2 * u_mesh_size_2 + ... + a_m * u_mesh_size_m) Tensor of ground truth evaluation values.
        self.f_meshes = data["f_meshes"] # List of length n_f_mesh; items are of size f_mesh_size_i x 2 Tensor of source term meshes.
        self.f_values = data["f_values"] # List of length n_f_mesh; items are of size num_expr * len(f_mesh_size) Tensor of source term values.
        self.u_length = len(self.evaluation_values) 
        self.f_length = len(self.f_values) 
        self.u_data_addresses = data["u_data_addresses"] # List of length n_f_mesh (number of f_meshes); each item contains n_expr (number of source terms) of tuples (start, end) for u_addresses for each source term.
        self.f_data_addresses = data["f_data_addresses"] # List of length n_f_mesh (number of f_meshes); each item contains n_expr (number of source terms) of tuples (start, end) for u_addresses for each source term.
        self.u_to_f_mesh_idx = data["u_to_f_mesh_idx"] # List of length u_length (number of source terms) containing the index of the f_mesh for each u coordinate point.
        self.u_point_to_expr_idx = data["u_point_to_expr_idx"] # List of length u_length (number of source terms) containing the index of the source term for each u coordinate point.
        self.mesh_addresses = data["mesh_size_addresses"] # List of length n_f_mesh of tuples (start, end) for the evaluation points for each corresponding mesh size.
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
            if subset_idx < self.u_length:
                idxs = random.sample(range(0, self.u_length), subset_idx)
                self.u_to_f_mesh_idx = [data["u_to_f_mesh_idx"][i] for i in idxs]
                sorted_indices = sorted(range(len(self.u_to_f_mesh_idx)),key=lambda i: self.u_to_f_mesh_idx [i])

                self.evaluation_mesh = self.evaluation_mesh[sorted_indices]
                self.evaluation_values = self.evaluation_values[sorted_indices]
                self.u_to_f_mesh_idx = [self.u_to_f_mesh_idx[i] for i in sorted_indices]
                self.u_point_to_expr_idx = [data["u_point_to_expr_idx"][i] for i in sorted_indices]
                self.u_length = subset_idx      
                mesh_address_start = 0
                new_mesh_addresses = []
                mesh_counter = Counter(self.u_to_f_mesh_idx)
                for key, value in sorted(mesh_counter.items()):
                    new_mesh_addresses.append((mesh_address_start, mesh_address_start + value))
                    mesh_address_start += value
                self.mesh_addresses = new_mesh_addresses
            elif subset_idx > self.u_length:
                raise ValueError(f"Subset index {subset_idx} is larger than the dataset length {self.u_length}.")




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

    # def get_f_mesh(self, u_idx: int | list[int]) -> list[torch.Tensor]:
    #     '''
    #     Returns the f_mesh for the given index of a u coordinate point.
    #     '''
    #     idx = self.u_to_f_mesh_idx[u_idx] # Get the index of the corresponding f_mesh for the given u coordinate point.
    #     if type(idx) == list:
    #         raise NotImplementedError("Not implemented yet for lists.")
    #         f_meshes = [self.f_meshes[i] for i in idx]
    #     else:
    #         f_meshes = self.f_meshes[idx]
        
    #     return f_meshes # (len(u_idx) x f_mesh_size x 2) Tensor of the source term mesh.
    
    # def get_f_values(self, u_idx: int | list[int]) -> list[torch.Tensor]:
    #     '''
    #     Returns the f_values for the given index of a u coordinate point.
    #     '''
    #     mesh_idx = self.u_to_f_mesh_idx[u_idx] # Get the index of the source term for the given u coordinate point.
    #     value_idx = self.u_point_to_expr_idx[u_idx] # Get the index of the source term for the given u coordinate point.

    #     if type(u_idx) == list:
    #         raise NotImplementedError("Not implemented yet for lists.")
    #         f_values = [self.f_values[idx][value_idx[i]] for i, idx in enumerate(mesh_idx)]
    #     else:
    #         f_values = self.f_values[mesh_idx][value_idx]
    #     return f_values # (len(u_idx) x f_mesh_size) Tensor of the source term mesh.

    def __len__(self):
        # return total dataset size
        return self.u_length

    def __getitem__(self, index) -> DatasetReturnClass:
        '''
        Returns the evaluation mesh, evaluation values, f_mesh_idx and f_values for the given index.
        We store the different f_meshes as a list of tensors as an attribute of the dataset.
        In this implementation, we assume that we return one single f_mesh_idx
        Using the GreenBatchSampler below, for each batch, we sample a random mesh size and return the corresponding f_mesh and f_values.
        '''
        if isinstance(index, int):
            ret_item = DatasetReturnClass(
                crd=self.evaluation_mesh[index],
                u_vals=self.evaluation_values[index],
                f_mesh_idx=self.u_to_f_mesh_idx[index],
                f_vals=self.f_values[self.u_to_f_mesh_idx[index]][self.u_point_to_expr_idx[index]]
            )
            return ret_item
        elif isinstance(index, slice):
            u_to_f_mesh_idx=self.u_to_f_mesh_idx[index]
            u_point_to_expr_idx=self.u_point_to_expr_idx[index]
            ret_item = DatasetReturnClass(
                crd=self.evaluation_mesh[index],
                u_vals=self.evaluation_values[index],
                f_mesh_idx=self.u_to_f_mesh_idx[index],
                f_vals=torch.stack([self.f_values[u_to_f_mesh_idx[i]][u_point_to_expr_idx[i]] for i in range(len(u_to_f_mesh_idx))])
            )
            return ret_item

        elif isinstance(index, list[slice]):
            total_crd = []
            total_u_vals = []
            total_f_mesh_idx = []
            total_f_vals = []
            for i in index:
                u_to_f_mesh_idx=self.u_to_f_mesh_idx[i]
                u_point_to_expr_idx=self.u_point_to_expr_idx[i]
                logger.info(u_to_f_mesh_idx, u_point_to_expr_idx)
                total_crd.append(self.evaluation_mesh[i])
                total_u_vals.append(self.evaluation_values[i])
                total_f_mesh_idx += self.u_to_f_mesh_idx[i]
                total_f_vals.append(torch.cat([self.f_values[u_to_f_mesh_idx[i]][u_point_to_expr_idx[i]] for i in range(len(u_to_f_mesh_idx))]))
            ret_item = DatasetReturnClass(
                crd=self.evaluation_mesh[index],
                u_vals=self.evaluation_values[index],
                f_mesh_idx=self.u_to_f_mesh_idx[index],
                f_vals=torch.stack([self.f_values[u_to_f_mesh_idx[i]][u_point_to_expr_idx[i]] for i in range(len(u_to_f_mesh_idx))])
            )
            return ret_item


class GreenBatchSampler(BatchSampler):
    '''
    BatchSampler that samples a random mesh size for each batch.
    It uses the mesh_size_addresses to sample a random mesh size, and from the chosen mesh size, 
        it samples a batch of corresponding indices within the range of idx addresses associated with that mesh size.
    '''
    def __init__(self, sampler, batchsize, drop_last, mesh_size_addresses):
        super().__init__(sampler, batchsize, drop_last)
        self.num_mesh_sizes = len(mesh_size_addresses)
        self.mesh_size_addresses = mesh_size_addresses
        self.batchsize = batchsize

    def __iter__(self):
        for _ in range(0, len(self.sampler), self.batchsize):   # indices from sampler
            i = random.randint(0, self.num_mesh_sizes - 1)
            mesh_size_address = self.mesh_size_addresses[i]
            assert mesh_size_address[1] - mesh_size_address[0] >= self.batchsize, \
                f"Mesh size address {mesh_size_address} is smaller than batch size {self.batchsize}."

            batch = random.sample(range(*mesh_size_address), self.batchsize)
            yield batch
        if len(batch) > 0 and not self.drop_last:
            yield batch


def greens_pinn_dataset_collate_fn(batch: list[DatasetReturnClass]) -> DatasetReturnClass:
    '''
    Collate function for the GreenPINNDataset.
    '''
    crd = torch.stack([item.crd for item in batch])
    u_vals = torch.stack([item.u_vals for item in batch])
    f_mesh_idx = batch[0].f_mesh_idx
    f_vals = torch.stack([item.f_vals for item in batch])
    
    return DatasetReturnClass(crd=crd, u_vals=u_vals, f_mesh_idx=f_mesh_idx, f_vals=f_vals)

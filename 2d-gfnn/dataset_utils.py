from dataclasses import dataclass, field
import math
from typing import Iterable, TypeVar, Generic
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler
from constants_utils import Hyperparameters
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
    def __init__(self, 
                 data_file_path: str, 
                 data_file_name: str, 
                 config: Hyperparameters, 
                 subset_idx: int = None):
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
        self.f_values = data["f_values"] # List of length n_f_mesh; items are of size num_expr x len(f_mesh_size) Tensor of source term values.
        self.u_length = len(self.evaluation_values) 

        # Addresses for individual f mesh sizes and f_values
        self.u_to_f_mesh_idx = data["u_to_f_mesh_idx"] # List of length u_length (number of source terms) containing the index of the f_mesh for each u coordinate point.
        self.u_point_to_expr_idx = data["u_point_to_expr_idx"] # List of length u_length (number of source terms) containing the index of the source term for each u coordinate point.
        self.u_data_addresses = data["u_data_addresses"] # List of length n_f_mesh (number of f_meshes); each item contains n_expr (number of source terms) of tuples (start, end) for u_addresses for each source term.
        self.mesh_addresses = data["mesh_size_addresses"] # List of length n_f_mesh of tuples (start, end) for the evaluation points for each corresponding mesh size.
        # Calculate starting indices in relation to amount of total source terms for each new size f - i.e. calculating n_expr_1 + n_expr_2 + ... + n_expr_n   
        self.num_f_terms = []
        start = 0
        for n in data["num_f_terms"]:
            self.num_f_terms.append(start)
            start += n

        self.config = config

        if self.config.multi_mesh_training_variant == "standardize":
            # Standardize f_meshes and f_values
            largest_integration_mesh_size = math.prod(self.constants.integration_mesh_sizes[-1])
            logger.info(f"Standardizing integration mesh sizes to size: {largest_integration_mesh_size}.")
            for i, (mesh, v) in enumerate(zip(self.f_meshes, self.f_values)):
                current_mesh_size = mesh.shape[0]
                self.f_meshes[i] = torch.nn.functional.pad(mesh, (0, 0, 0, largest_integration_mesh_size-current_mesh_size), mode="constant", value=-1)
                self.f_values[i] = torch.nn.functional.pad(v, (0, largest_integration_mesh_size-current_mesh_size, 0, 0), mode="constant", value=-1)
            
            self.f_meshes = torch.stack(self.f_meshes)
            self.f_values = torch.stack(self.f_values)
            #

        if subset_idx is not None:
            # subset_idx = 20
            if subset_idx < self.u_length: 

                # Get subset_idx amount of random indices. 
                idxs = random.sample(range(0, self.u_length), subset_idx)
                self.u_length = subset_idx

                # Need to sort indices by their f_mesh_idx and then u_point_to_expr_idx to ensure that mesh_addresses works properly.
                sorted_indices = sorted(idxs, key=lambda idx: self.num_f_terms[self.u_to_f_mesh_idx[idx]] + self.u_point_to_expr_idx[idx])

                # Get subset points and values
                self.evaluation_mesh = self.evaluation_mesh[sorted_indices]
                self.evaluation_values = self.evaluation_values[sorted_indices]

                # Change address pointers
                self.u_to_f_mesh_idx = [self.u_to_f_mesh_idx[i] for i in sorted_indices]
                self.u_point_to_expr_idx = [self.u_point_to_expr_idx[i] for i in sorted_indices]

                # Set Counter for easier computing of number of elements per f_mesh.
                f_mesh_counter = Counter(self.u_to_f_mesh_idx)

                # Update num_f_terms with the new amount of points.
                new_num_f_terms = []
                num_f_term_start = 0
                for key, value in f_mesh_counter.items():
                    new_num_f_terms.append(num_f_term_start)
                    num_f_term_start += value
                self.num_f_terms = new_num_f_terms

                # Update f_meshes to only include the meshes that were still picked.
                self.f_meshes = torch.stack([self.f_meshes[key] for key in f_mesh_counter.keys()])

                # Update f_values to only include the meshes that were still picked.

                # First isolate the f_mesh associated sub-lists that are actually included in the datset.

                # Store f_values associated with an f_mesh in dictionary
                new_f_values_dict = {idx : [] for idx in f_mesh_counter.keys()}
                for f_mesh_idx, expr_idx in zip(self.u_to_f_mesh_idx, self.u_point_to_expr_idx):
                    new_f_values_dict[f_mesh_idx].append(self.f_values[f_mesh_idx][expr_idx])
                
                new_f_values = []

                # Used for case of standardize variant 
                max_n_f_terms = max(f_mesh_counter.values())
                for _, v in new_f_values_dict.items():
                    if self.config.multi_mesh_training_variant == "standardize":
                        # Pad out values so that each f_mesh associated f_values tensor has the same size. E.g. if some f_mesh only has 2 f_value tensors, and another has 3, we can't stack these tensors.
                        v = torch.nn.functional.pad(torch.stack(v), (0, 0, 0, max_n_f_terms-len(v)), mode="constant", value=0)
                    new_f_values.append(v)


                # Use Counter to count number of elements associated with one mesh size.
                mesh_address_start = 0
                new_mesh_addresses = []
                for key, value in sorted(f_mesh_counter.items()):
                    new_mesh_addresses.append((mesh_address_start, mesh_address_start + value))
                    mesh_address_start += value
                self.mesh_addresses = new_mesh_addresses

                # Update indices counter to change from global representation (e.g. index 10 is the 10th in relation to the entire dataset) 
                #   to new local representation (i.e. now we only have a select f_meshes and f_values, so index 10 would refer to the 10th element in relation to the new subset.)
                local_u_to_f_mesh_idx = []
                local_u_point_to_expr_idx = []

                for i, (n, mesh_address) in enumerate(zip(f_mesh_counter.values(), self.mesh_addresses)):
                    local_u_to_f_mesh_idx.extend([i]*n)
                    current_f_mesh_source_term_counter = Counter(self.u_point_to_expr_idx[slice(*mesh_address)])
                    for j, (k, v) in enumerate(current_f_mesh_source_term_counter.items()):
                        local_u_point_to_expr_idx.extend([j]*v)
                self.u_to_f_mesh_idx = local_u_to_f_mesh_idx
                self.u_point_to_expr_idx = local_u_point_to_expr_idx

                # Update u_data_addresses.
                new_u_data_addresses = []
                for mesh_address in self.mesh_addresses:
                    sub_indices = sorted_indices[slice(*mesh_address)]

                    # Count the amount of elements that have a specific source term index
                    source_term_counter = Counter(sub_indices)
                    u_data_address_start = 0
                    f_mesh_u_data_address = [] 
                    for _, value in source_term_counter.items():
                        f_mesh_u_data_address.append((u_data_address_start, u_data_address_start + value))
                        u_data_address_start += value
                    new_u_data_addresses.append(f_mesh_u_data_address)

                self.u_data_addresses = new_u_data_addresses
                
                    
                    


            elif subset_idx > self.u_length:
                raise ValueError(f"Subset index {subset_idx} is larger than the dataset length {self.u_length}.")
            else:
                logger.info("Subset_idx param is same size as dataset, will not create subset of data.")




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
        # logger.info(f"{index}, {self.u_to_f_mesh_idx[index]}, {self.u_point_to_expr_idx[index]}")
        if isinstance(index, int):
            ret_item = DatasetReturnClass(
                crd=self.evaluation_mesh[index],
                u_vals=self.evaluation_values[index],
                f_mesh_idx=self.u_to_f_mesh_idx[index],
                f_vals=self.f_values[self.u_to_f_mesh_idx[index]][self.u_point_to_expr_idx[index]]
            )
            return ret_item
        elif isinstance(index, slice):
            assert self.config.multi_mesh_training_variant == "standardize", "Currently only implemented for standardize because of the f_vals stack function."
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
            assert self.config.multi_mesh_training_variant == "standardize", "Currently only implemented for standardize because of the f_vals stack function."
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


class GreenOneByOneBatchSampler(BatchSampler):
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


def greens_pinn_dataset_total_collate_fn(batch: list[DatasetReturnClass]) -> DatasetReturnClass:
    '''
    Collate function for the GreenPINNDataset in one by one fashion.
    '''
    crd = torch.stack([item.crd for item in batch])
    u_vals = torch.stack([item.u_vals for item in batch])
    f_mesh_idx = [item.f_mesh_idx for item in batch]
    f_vals = torch.stack([item.f_vals for item in batch])
    
    return DatasetReturnClass(crd=crd, u_vals=u_vals, f_mesh_idx=f_mesh_idx, f_vals=f_vals)

def greens_pinn_dataset_obo_collate_fn(batch: list[DatasetReturnClass]) -> DatasetReturnClass:
    '''
    Collate function for the GreenPINNDataset in one by one fashion.
    '''
    crd = torch.stack([item.crd for item in batch])
    u_vals = torch.stack([item.u_vals for item in batch])
    f_mesh_idx = batch[0].f_mesh_idx
    f_vals = torch.stack([item.f_vals for item in batch])
    
    return DatasetReturnClass(crd=crd, u_vals=u_vals, f_mesh_idx=f_mesh_idx, f_vals=f_vals)

from dataclasses import dataclass, field
import torch
from torch.utils.data import Dataset
from plot_utils import plot_points
from constants_utils import mesh_type
from loss import fetch_quadrature_weights
from data_generation_utils import gcd_chebyshev_mesh_size, sample_points



def get_interior_boundary_idx(domain, mesh):
    '''
    Returns the indices of collocation and boundary points in the given points.

    :param tuple domain: (x_min, x_max, y_min, y_max) tuple.
    :param Tensor mesh: b x 2 Tensor.

    :return interior_ind: (intr x 2) Tensor
    :return boundary_ind: ((b - intr) x 2) Tensor
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

    :return non_corners_idx: (n_c x 2) Tensor
    :return corners_idx: ((b - n_c) x 2) Tensor
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

    Args:
        chebyshev_evaluation_mesh: Experimental parameter that can be used to later interpolate values from the values on the mesh.
    '''
    domain: tuple
    integration_mesh: torch.Tensor
    evaluation_mesh_type: mesh_type
    integration_mesh_type: mesh_type
    evaluation_mesh_size: tuple
    integration_mesh_size: tuple
    chebyshev_evaluation_mesh: torch.Tensor = field(init=False)
    quadrature_weights: torch.Tensor = field(init=False)

    def __post_init__(self):
        if not self.evaluation_mesh_type == "chebyshev":
            chebyshev_mesh_size = gcd_chebyshev_mesh_size(self.integration_mesh_size)
            self.chebyshev_evaluation_mesh = sample_points(self.domain, chebyshev_mesh_size)
        else: 
            self.chebyshev_evaluation_mesh = sample_points(self.domain, self.evaluation_mesh_size)

        self.quadrature_weights = fetch_quadrature_weights(domain=self.domain, 
                                                           integration_mesh_size=self.integration_mesh_size, 
                                                           integration_mesh_type=self.integration_mesh_type)

class GreenPINNDataset(Dataset):
    '''
    Wrapper to retrieve all datasets and store them into one main dataset wrapper.
    All datasets are concatenated with each other.
    We use a pointer system for mapping u_values with their corresponding f_meshes and f_values to avoid duplicate copies of the data.
    '''
    def __init__(self, data_file_path, data_file_name: str, ):
        self.data = torch.load(data_file_path + data_file_name)

        self.constants = GreensConstantsDataclass(
            domain=self.data["domain"],
            integration_mesh=self.data["f_meshes"][0],
            evaluation_mesh_type=self.data["u_mesh_type"],
            integration_mesh_type=self.data["f_mesh_type"],
            evaluation_mesh_size=self.data["u_mesh_size"],
            integration_mesh_size=self.data["f_mesh_size"]
        )
        
        self.length = len(self.data["u_values"])
        self.evaluation_mesh = self.data["coordinates"]
        self.evaluation_values = self.data["u_values"]
        self.f_values = self.data["f_values"]
        self.f_meshes = self.data["f_meshes"]

        self.f_inds = [0] * self.length
        self.sub_lengths = self.data["data_addresses"]
        for i, address in enumerate(self.data["data_addresses"]):
            self.f_inds[slice(*address)] = [i] * (address[1]-address[0])


    def interior_points_dataset(self):
        '''
        Modify the dataset to only use interior points in the evaluation mesh, exclude the boundary points.
        '''
        intr, _ = get_interior_boundary_idx(domain=self.constants.domain, mesh=self.evaluation_mesh)
        self.evaluation_mesh = self.evaluation_mesh[intr]
        self.evaluation_values = self.evaluation_values[intr]
        self.length = len(self.evaluation_mesh)
        self.f_inds = [self.f_inds[i] for i in intr]

        ##Temporary solution to fix the sub_lengths for interior points.
        sub_length = self.length // len(self.sub_lengths)
        for i in range(len(self.sub_lengths)):
            self.sub_lengths[i] = (i*sub_length, (i+1)*sub_length)

    def exclude_corners_dataset(self):
        '''
        Modify the dataset to exclude corner points in the evaluation mesh. 
        '''
        non_corners_idx, _ = get_corners_idx(domain=self.domain, mesh=self.evaluation_mesh)
        self.evaluation_mesh = self.evaluation_mesh[non_corners_idx]
        print(non_corners_idx.shape, self.evaluation_mesh.shape)
        self.evaluation_values = self.evaluation_values[non_corners_idx]
        self.length = len(self.evaluation_mesh)
        self.f_inds = [self.f_inds[i] for i in non_corners_idx]

        ##Temporary solution to fix the sub_lengths for interior points.
        sub_length = self.length // len(self.sub_lengths)
        for i in range(len(self.sub_lengths)):
            self.sub_lengths[i] = (i*sub_length, (i+1)*sub_length)

    def plot_evaluation_mesh(self):
        '''
        Debug tool to plot evaluation mesh.
        '''
        plot_points(points=self.evaluation_mesh[slice(*self.sub_lengths[0])], values=self.evaluation_values[slice(*self.sub_lengths[0])])

    def __len__(self):
        # return total dataset size
        return self.length

    def __getitem__(self, index):
        '''
        Returns
        '''
        ret_item = {"crd": self.evaluation_mesh[index], "u_vals": self.evaluation_values[index], "f_inds": self.f_inds[index],
                    "f_vals": self.f_values[self.f_inds[index]], "f_mesh": self.f_meshes[self.f_inds[index]]}
        return ret_item
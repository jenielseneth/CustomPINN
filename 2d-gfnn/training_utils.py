import math
from torch.utils.data import Dataset, DataLoader
from data_generation_utils import sample_chebyshev_points_3
from plot_utils import plot_multiple_points
import torch
import random
import os

from loss import DataPredLoss


class DatasetWrapper(Dataset):
    def __init__(self, file_path: str):
        self.data = torch.load(file_path)
        self.coordinates = self.data["coordinates"]
        self.u_values = self.data["u_values"]
        self.f_values = self.data["f_values"]
        self.f_mesh = self.data["f_mesh"]
        self.length = len(self.coordinates)
        # load the images from file

    def __len__(self):
        # return total dataset size
        return self.length

    def __getitem__(self, index):
        # write your code to return each batch element
        return self.coordinates[index], self.u_values[index], self.f_values, self.f_mesh


def fetch_dataset(file_path: str, data_file_path: str):
    data = torch.load(file_path + data_file_path)
    coordinates = data["coordinates"]
    u_values = data["u_values"]
    f_values = data["f_values"]
    f_mesh = data["f_mesh"]
    return coordinates, u_values, f_values, f_mesh

class UpdatedMultiDatasetWrapper(Dataset):
    '''
    Wrapper to retrieve all datasets and store them into one main dataset wrapper.
    All datasets are concatenated with each other.
    We use a pointer system for mapping u_values with their corresponding f_meshes and f_values to avoid duplicate copies of the data.
    '''
    def __init__(self, data_file_path, data_file_name: str, domain: tuple, interior: bool = True):
        self.domain = domain
        self.data = torch.load(data_file_path + data_file_name)
        self.length = self.data["coordinates"].shape[0]
        self.coordinates = self.data["coordinates"]
        self.u_values = self.data["u_values"]
        self.f_values = self.data["f_values"]
        self.f_meshes = self.data["f_meshes"]
        self.interior_idxs = self.data["interior_idxs"]
        self.boundary_idxs = self.data["boundary_idxs"]

        self.f_inds = [0] * self.length
        self.sub_lengths = self.data["data_addresses"]
        for i, address in enumerate(self.data["data_addresses"]):
            self.f_inds[slice(*address)] = [i] * (address[1]-address[0])
        

        self.u_mesh_type = self.data["u_mesh_type"]
        self.u_mesh_size = self.data["u_mesh_size"]
        self.f_mesh_type = self.data["f_mesh_type"]
        self.f_mesh_size = self.data["f_mesh_size"]

        if interior:
            self.coordinates = self.coordinates[self.interior_idxs]
            self.u_values = self.u_values[self.interior_idxs]
            self.f_inds = [self.f_inds[i] for i in self.interior_idxs]
            self.length = len(self.coordinates)

            ##Temporary solution to fix the sub_lengths for interior points.
            sub_length = self.length // len(self.sub_lengths)
            for i in range(len(self.sub_lengths)):
                self.sub_lengths[i] = (i*sub_length, (i+1)*sub_length)

    def __len__(self):
        # return total dataset size
        return self.length

    def __getitem__(self, index):
        # write your code to return each batch element

        # write your code to return each batch element
        ret_item = {"crd": self.coordinates[index], "u_vals": self.u_values[index], "f_inds": self.f_inds[index],
                    "f_vals": self.f_values[self.f_inds[index]], "f_mesh": self.f_meshes[self.f_inds[index]]}
        return ret_item


def train(model, optimizer, dataloader: DataLoader, loss_fn: DataPredLoss, 
          bnd_loss: bool, boundary_points, domain_mesh):
    '''
    boundary_points: bnd x dom_mesh x 2 Tensor of boundary points to evaluate the boundary loss on.
    domain_mesh: bnd x dom_mesh x 2 Tensor of domain mesh points to evaluate the boundary loss on.
    bnd_loss: bool, whether to include the boundary loss in the training.
    '''
    size = len(dataloader.dataset)
    model.train()
    current_num = 0
    total_loss = 0
    
    for i, item in enumerate(dataloader):
        # Compute prediction and loss
            bs = len(item["crd"])
            loss = loss_fn(greens_function_approx=model, f_values_batch=item["f_vals"],
                        f_meshes_batch=item["f_mesh"], coordinates_batch=item["crd"],  u_batch=item["u_vals"])

            if bnd_loss:
                assert boundary_points.shape == domain_mesh.shape, f"Boundary points ({boundary_points.shape}) and domain mesh ({domain_mesh.shape}) must have the same batch size."
                bnd_eval = model(boundary_points, domain_mesh)
                bnd_loss = torch.nn.functional.mse_loss(bnd_eval, torch.zeros_like(bnd_eval))
                loss += bnd_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()
            current_num += bs
            print(f"\rAvg Train Loss per sample: {loss / bs :>7f}  [{current_num:>5d}/{size:>5d}] \n", end="")
            total_loss += loss

    return total_loss


def test(dataloader, model, loss_fn: DataPredLoss, domain):
    size = len(dataloader.dataset)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for item in dataloader:
            loss = loss_fn(greens_function_approx=model, f_values_batch=item["f_vals"],
                        f_meshes_batch=item["f_mesh"], coordinates_batch=item["crd"],  u_batch=item["u_vals"])
            test_loss += loss.item()

    print(f"Avg Test Loss per sample: {test_loss/ size :>8f} \n", end="")

    return test_loss



import math
from torch.utils.data import Dataset, DataLoader
import torch
import random
import os

from loss3d import CustomDataPredLoss3d


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
    

# class WholeDatasetWrapper(Dataset):
#     def __init__(self, collocation_file_path: str, boundary_file_path: str):
#         self.col_data = torch.load(collocation_file_path)
#         self.bnd_data = torch.load(boundary_file_path)
#         self.c_coordinates = self.col_data["coordinates"]
#         self.c_values = self.col_data["values"]
#         self.b_coordinates = self.bnd_data["coordinates"]
#         self.b_values = self.bnd_data["values"]
#         self.coordinates = torch.cat((self.c_coordinates, self.b_coordinates))
#         self.values = torch.cat((self.c_values, self.b_values))
#         self.length = len(self.coordinates)

#     def __len__(self):
#         # return total dataset size
#         return self.length

#     def __getitem__(self, index):
#         # write your code to return each batch element
#         return self.coordinates[index], self.values[index]


def fetch_dataset(file_path: str, data_file_path: str):
    data = torch.load(file_path + data_file_path)
    coordinates = data["coordinates"]
    u_values = data["u_values"]
    f_values = data["f_values"]
    f_mesh = data["f_mesh"]
    return coordinates, u_values, f_values, f_mesh

class MultiDatasetWrapper(Dataset):

    def __init__(self, data_file_path: str,data_file_name: str):
        if data_file_path[-1] != "/":
            data_file_path += "/"
        subdirectories = [data_file_path + a + "/" for a in os.listdir(data_file_path) if os.path.isdir(data_file_path + a)]
        print(subdirectories)
        c, u, f, lengths, f_meshes = [], [], [], [], []
        for i, subdir in enumerate(subdirectories):
            coordinates, u_values, f_values, f_mesh = fetch_dataset(subdir, data_file_name)
            c.append(coordinates)
            u.append(u_values)
            f.append(f_values)
            f_meshes.append(f_mesh)
            lengths.append(len(coordinates))

        self.coordinates = torch.cat(c)
        self.u_values = torch.cat(u)
        self.f_values = f
        self.f_meshes = f_meshes
        self.total_length = len(self.coordinates)
        self.f_inds = [0] * self.total_length
        self.sub_lengths = lengths
        self.start_inds = [0] * len(f_meshes)
        start = 0
        for i, length in enumerate(lengths):
            self.f_inds[start:start+length] = [i] * length
            start +=length
            if i!=0:
                self.start_inds[i] = self.start_inds[i-1]+lengths[i-1]

    def __len__(self):
        # return total dataset size
        return self.total_length

    def __getitem__(self, index):
        # write your code to return each batch element
        return self.coordinates[index], self.u_values[index], self.f_inds[index]


def train(model, optimizer, dataloader: DataLoader, loss_fn: CustomDataPredLoss3d, 
          f_values, f_meshes, domain, scheduler = None):
    size = len(dataloader.dataset)
    model.train()
    current_num = 0
    total_loss = 0
    for i, (coordinates, u_values, f_inds) in enumerate(dataloader):
        # Compute prediction and loss
        loss = loss_fn(greens_function_approx=model, domain=domain, f_values=f_values, 
                       f_meshes=f_meshes, coordinates=coordinates, f_inds=f_inds, u=u_values)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss = loss.item()
        current_num= len(coordinates) + current_num
        print(f"\rAvg Train Loss per sample: {loss:>7f}  [{current_num:>5d}/{size:>5d}] \n", end="")
        total_loss += loss
    
    return total_loss

def test(dataloader, model, loss_fn: CustomDataPredLoss3d, f_values, f_meshes, domain):
    size = len(dataloader.dataset)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for coordinate, u_value, f_inds in dataloader:
            test_loss += loss_fn(greens_function_approx=model, domain=domain, f_values=f_values, f_meshes=f_meshes,
            coordinates=coordinate, f_inds=f_inds, u=u_value).item()

    print(f"Avg Test Loss per sample: {test_loss/ len(dataloader) :>8f} \n", end="")

    return test_loss



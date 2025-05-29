import math
from torch.utils.data import Dataset, DataLoader
import torch
import random
import os

from loss import UpdatedDataPredLoss


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
    

def get_collocation_boundary_idx(domain, points):
    boundary_ind = []
    collocation_ind = []
    for i, point in enumerate(points):
        if point[0] in domain[0:2] or point[1] in domain[2:4]:
            boundary_ind.append(i)
        else:
            collocation_ind.append(i)
    return collocation_ind, boundary_ind
    boundary_data = points[boundary_ind]
    collocation_data = points[collocation_ind]
    return collocation_data, boundary_data

def fetch_dataset(file_path: str, data_file_path: str):
    data = torch.load(file_path + data_file_path)
    coordinates = data["coordinates"]
    u_values = data["u_values"]
    f_values = data["f_values"]
    f_mesh = data["f_mesh"]
    return coordinates, u_values, f_values, f_mesh

class MultiDatasetWrapper(Dataset):

    def __init__(self, data_file_path: str,data_file_name: str, domain: tuple):
        self.domain = domain
        if data_file_path[-1] != "/":
            data_file_path += "/"
        subdirectories = [data_file_path + a + "/" for a in os.listdir(data_file_path) if os.path.isdir(data_file_path + a)]
        # print(subdirectories)
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


class MultiBndDatasetWrapper(Dataset):

    def __init__(self, data_file_path: str,data_file_name: str, domain: tuple):
        self.domain = domain

        if data_file_path[-1] != "/":
            data_file_path += "/"

        ## Sort subdirectories by index as generated using generate_data.py
        subdirectories = [data_file_path + a + "/" for a in os.listdir(data_file_path) if os.path.isdir(data_file_path + a)]
        subdirs_int = [int(a) for a in os.listdir(data_file_path) if os.path.isdir(data_file_path + a)]
        subdirectories = [x for _ , x in sorted(zip(subdirs_int,subdirectories))]

        c, u, c_bnd, u_bnd, f, f_meshes = [], [], [], [], [], []
        for i, subdir in enumerate(subdirectories):
            coordinates, u_values, f_values, f_mesh = fetch_dataset(subdir, data_file_name)
            col_idx, bnd_idx = get_collocation_boundary_idx(self.domain, coordinates)
            c.append(coordinates[col_idx])
            c_bnd.append(coordinates[bnd_idx])
            u.append(u_values[col_idx])
            u_bnd.append(u_values[bnd_idx])
            f.append(f_values)
            f_meshes.append(f_mesh)
        self.coordinates = torch.stack(c)
        self.bnd_coordinates = torch.stack(c_bnd)
        self.u_values = torch.stack(u)
        self.u_bnd_values = torch.stack(u_bnd)
        self.f_values = f
        self.f_meshes = f_meshes
        self.total_length = len(self.coordinates)

    def __len__(self):
        # return total dataset size
        return self.total_length

    def __getitem__(self, index):
        # write your code to return each batch element
        ret_item = {"col_crd": self.coordinates[index], "col_u_vals": self.u_values[index], "bnd_crd": self.bnd_coordinates[index], 
                    "bnd_u_vals": self.u_bnd_values[index], "f_vals": self.f_values[index], "f_mesh": self.f_meshes[index]}
        return ret_item

# def train(model, optimizer, dataloader: DataLoader, loss_fn: DataPredLoss, 
#           f_values, f_meshes, domain, scheduler = None):
#     size = len(dataloader.dataset)
#     model.train()
#     current_num = 0
#     total_loss = 0
#     for i, (coordinates, u_values, f_inds) in enumerate(dataloader):
#         # Compute prediction and loss
#         loss = loss_fn(greens_function_approx=model, domain=domain, f_values=f_values, 
#                        f_meshes=f_meshes, f_inds=f_inds,
#                        coordinates=coordinates,  u=u_values)
#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if scheduler is not None:
#             scheduler.step()

#         loss = loss.item()
#         current_num= len(coordinates) + current_num
#         print(f"\rAvg Train Loss per sample: {loss:>7f}  [{current_num:>5d}/{size:>5d}] \n", end="")
#         total_loss += loss
    
#     return total_loss

# def test(dataloader, model, loss_fn: DataPredLoss, f_values, f_meshes, domain):
#     size = len(dataloader.dataset)
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for coordinate, u_value, f_inds in dataloader:
#             test_loss += loss_fn(greens_function_approx=model, domain=domain, f_values=f_values, f_meshes=f_meshes,
#             coordinates=coordinate, f_inds=f_inds, u=u_value).item()

#     print(f"Avg Test Loss per sample: {test_loss/ len(dataloader) :>8f} \n", end="")

#     return test_loss

def train_2(model, optimizer, dataloader: DataLoader, loss_fn: UpdatedDataPredLoss, 
            domain, scheduler = None):
    size = len(dataloader.dataset)
    model.train()
    current_num = 0
    total_loss = 0
    for i, item in enumerate(dataloader):
        # Compute prediction and loss
            col_loss = loss_fn(greens_function_approx=model, domain=domain, f_values_batch=item["f_vals"], 
                        f_mesh_batch=item["f_mesh"], coordinates_batch=item["col_crd"],  u_batch=item["col_u_vals"])
            bnd_loss = loss_fn(greens_function_approx=model, domain=domain, f_values_batch=item["f_vals"], 
                        f_mesh_batch=item["f_mesh"], coordinates_batch=item["bnd_crd"],  u_batch=torch.zeros_like(item["bnd_u_vals"]))
            loss = col_loss + bnd_loss
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            loss = loss.item()
            current_num += 1
            print(f"\rAvg Train Loss per sample: {loss:>7f}  [{current_num:>5d}/{size:>5d}] \n", end="")
            total_loss += loss
    
    return total_loss

def test_2(dataloader, model, loss_fn: UpdatedDataPredLoss, domain):
    size = len(dataloader.dataset)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for item in dataloader:
            col_loss = loss_fn(greens_function_approx=model, domain=domain, f_values_batch=item["f_vals"], 
                        f_mesh_batch=item["f_mesh"], coordinates_batch=item["col_crd"],  u_batch=item["col_u_vals"])
            bnd_loss = loss_fn(greens_function_approx=model, domain=domain, f_values_batch=item["f_vals"], 
                        f_mesh_batch=item["f_mesh"], coordinates_batch=item["bnd_crd"],  u_batch=torch.zeros_like(item["bnd_u_vals"]))
            loss = (col_loss + bnd_loss).item()
            test_loss += loss

    print(f"Avg Test Loss per sample: {test_loss/ size :>8f} \n", end="")

    return test_loss



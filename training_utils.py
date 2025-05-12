import math
from torch.utils.data import Dataset, DataLoader
import torch
import random

from loss import CustomDataPredLoss


class DatasetWrapper(Dataset):
    def __init__(self, file_path: str):
        self.data = torch.load(file_path)
        self.coordinates = self.data["coordinates"]
        self.values = self.data["values"]
        self.length = len(self.coordinates)
        # load the images from file

    def __len__(self):
        # return total dataset size
        return self.length

    def __getitem__(self, index):
        # write your code to return each batch element
        return self.coordinates[index], self.values[index]
    

class WholeDatasetWrapper(Dataset):
    def __init__(self, colocation_file_path: str, boundary_file_path: str):
        self.col_data = torch.load(colocation_file_path)
        self.bnd_data = torch.load(boundary_file_path)
        self.c_coordinates = self.col_data["coordinates"]
        self.c_values = self.col_data["values"]
        self.b_coordinates = self.bnd_data["coordinates"]
        self.b_values = self.bnd_data["values"]
        self.coordinates = torch.cat((self.c_coordinates, self.b_coordinates))
        self.values = torch.cat((self.c_values, self.b_values))
        self.length = len(self.coordinates)
        print(self.length)
        # load the images from file

    def __len__(self):
        # return total dataset size
        return self.length

    def __getitem__(self, index):
        # write your code to return each batch element
        return self.coordinates[index], self.values[index]
class BoundaryDataloaderWrapper:
    def __init__(self, file_path: str, batch_size):
        self.data = torch.load(file_path)
        shuffle_ind = torch.randperm(len(self.data["coordinates"]))
        self.batch_size = batch_size
        self.coordinates = self.data["coordinates"][shuffle_ind]
        self.values = self.data["values"][shuffle_ind]
        self.length = math.ceil(len(self.data["coordinates"])/self.batch_size)
        self.split_coordinates = torch.split(self.coordinates, self.length)
        self.split_values = torch.split(self.values, self.length)
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # write your code to return each batch element
        return self.split_coordinates[index], self.split_values[index]

def train(model, optimizer, dataloader: DataLoader, loss_fn: CustomDataPredLoss, f_source_term, domain, scheduler = None):
    size = len(dataloader.dataset)
    model.train()
    current_num = 0
    for _, (coordinate, value) in enumerate(dataloader):
        # Compute prediction and loss
        loss = loss_fn(greens_function_approx=model, domain=domain, f_source_term=f_source_term, coordinates=coordinate, u=value)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss = loss.item()
        current_num= len(coordinate) + current_num
        print(f"\rAvg Train Loss per sample: {loss:>7f}  [{current_num:>5d}/{size:>5d}] \n", end="")

def test(dataloader, model, loss_fn: CustomDataPredLoss,  f_source_term, domain):
    size = len(dataloader.dataset)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for coordinate, value in dataloader:
            test_loss +=  loss_fn(greens_function_approx=model, f_source_term=f_source_term, coordinates=coordinate, domain=domain,u=value).item() * len(coordinate)

    print(f"Avg Test Loss per sample: {test_loss / size :>8f} \n", end="")
    return test_loss



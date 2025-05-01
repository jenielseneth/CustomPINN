####Get Dataset for training analytically 
from typing import Tuple
import torch
from torch import nn
import numpy as np
import scipy.integrate as integrate
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
##2D example

class PINN_NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PINN_NN, self).__init__()
        self.layers = nn.ModuleList()
        self.layer_num = 8
        for i in range(self.layer_num):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_size))
            elif i == self.layer_num-1:
                self.layers.append(nn.Linear(hidden_size, output_size))
            else:
                self.layers.append(nn.Linear(hidden_size, hidden_size))
                

    def forward(self, x, y):
        input = torch.cat((x, y), dim=1)
        for i, layer in enumerate(self.layers):
            input = layer(input)
            if i < len(self.layers) - 1:
                input = torch.relu(input)
        return input
    
def sample_mesh_points(x_min, x_max, y_min, y_max, num_points):
    x = np.random.uniform(x_min, x_max, num_points)
    y = np.random.uniform(y_min, y_max, num_points)
    return np.vstack((x, y)).T

def greens_function_poisson_eq_2d(x, y, center: Tuple[float, float]):
    """
    Greens function for the Poisson equation in 2D with Dirichlet boundary conditions.
    The function is defined as:
    G(x, y) = 1/(2*pi) * log(r), r < radius
    """
    r = ((x - center[0])**2 + (y - center[1])**2)
    G = -1/(2*np.pi) * np.log(r)
    return G

def test_source_term(x, y):
    """
    Test source term for the Poisson equation.
    """
    return np.sin(x) * np.cos(y)

def evaluate_u(x, y, domain: Tuple[float, float, float, float]):
    """
    Evaluate the solution u at the given points (x, y) using the Green's function.
    """
    u = integrate.dblquad(
        lambda i, j: greens_function_poisson_eq_2d(x, y, (i, j)) * test_source_term(i, j),
        domain[0], domain[1],
        lambda x: domain[2],
        lambda x: domain[3]
    )
    return u

class CustomPINN_Green2D(nn.Module):
    def __init__(self, dims: int, hidden_size: int = 64):
        self.phi = PINN_NN(input_size=dims, hidden_size=hidden_size, output_size=dims)
        self.psi = PINN_NN(input_size=dims, hidden_size=hidden_size, output_size=dims)

    def forward(self, x, y):
        return self.phi(x,y) * torch.log(x-y) + self.psi(x,y)

def generate_points(x_min, x_max, y_min, y_max, num_points):
    # Sample mesh points
    mesh_points = sample_mesh_points(x_min, x_max, y_min, y_max, num_points)
    
    # Define the domain for integration
    domain = (x_min, x_max, y_min, y_max)
    
    # Evaluate the solution u at the sampled points
    u_values = np.zeros(mesh_points.shape[0])
    for i in tqdm(range(mesh_points.shape[0])):
        u = evaluate_u(mesh_points[i, 0], mesh_points[i, 1], domain)
        u_values[i] = u[0]
    
    torch.save(torch.tensor(u_values), "u_values.pt")

def train(model, optimizer, dataloader, loss_fn, scheduler = None):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    current_num = 0
    for batch, (x, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss = loss.item()
        total_loss += loss
        current_num= len(x) + current_num
        print(f"\rloss: {loss:>7f}  [{current_num:>5d}/{size:>5d}] \n", end="")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x)
            test_loss += loss_fn(pred.real, y).item()

    print(f"Avg Test loss per sample: {test_loss /size:>8f} , Avg Test loss per batch: {test_loss /num_batches:>8f} \n", end="")
    return test_loss

class TrainingDataset(Dataset):
    def __init__(self):
        self.data = torch.load('u_values_train.pt')
        self.x = self.data[:][0]
        self.y = self.data[:][1]
        self.length = len(self.data)
        # load the images from file

    def __len__(self):
        # return total dataset size
        return self.length

    def __getitem__(self, index):
        # write your code to return each batch element
        return self.x[index], self.y[index]
    
class TestDataset(Dataset):
    def __init__(self):
        self.data = torch.load('u_values_test.pt')
        self.x = self.data[:][0]
        self.y = self.data[:][1]
        self.length = len(self.data)
        # load the images from file

    def __len__(self):
        # return total dataset size
        return self.length

    def __getitem__(self, index):
        # write your code to return each batch element
        return self.x[index], self.y[index]

if __name__ == "__main__":
    x_min, x_max = 0, 1 
    y_min, y_max = 0, 1
    num_points = 400 
    generate_points()
    torch.load("u_values.pt")



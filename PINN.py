
import torch.nn as nn
import torch

class PINN_NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PINN_NN, self).__init__()
        self.layers = nn.ModuleList()
        self.layer_num = 5
        for i in range(self.layer_num):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_size))
            elif i == self.layer_num-1:
                self.layers.append(nn.Linear(hidden_size, output_size))
            else:
                self.layers.append(nn.Linear(hidden_size, hidden_size))
                

    def forward(self, x, y):
        input = torch.cat((x,y), dim=-1)
        for i, layer in enumerate(self.layers):
            input = layer(input)
            if i < len(self.layers) - 1:
                input = torch.relu(input)
        return input
    
class CustomPINN_Green2D(nn.Module):
    def __init__(self, dims: int, output_size: int, hidden_size: int):        
        super(CustomPINN_Green2D, self).__init__()
        self.phi = PINN_NN(input_size=dims, hidden_size=hidden_size, output_size=output_size)
        self.psi = PINN_NN(input_size=dims, hidden_size=hidden_size, output_size=output_size)

    def forward(self, x, y):
        return self.phi(x,y) * torch.log(torch.abs(x-y).sum(-1)) + self.psi(x,y)

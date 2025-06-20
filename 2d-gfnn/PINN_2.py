
import torch.nn as nn
import torch

class PINN_NN_2(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(PINN_NN_2, self).__init__()
        self.feature_extractor = MLP(input_size=2, hidden_size=hidden_size, output_size=hidden_size, num_layers=num_layers)

    def forward(self, x, y):
        '''
        x is the input coordinate for u(x) = int (G(x,y) * f(y) dy).
        y is the parameter along which we integrate.
        x: b x f x 2 Tensor; b - batch size of coordinates, f - size of f_mesh, 2 - 2D
        y: b x f x 2 Tensor; b - batch size of coordinates, f - size of f_mesh, 2 - 2D
        '''
        x_features = self.feature_extractor(x) # b x f x output_size 
        y_features = self.feature_extractor(y) # b x f x output_size 
        # print(x_features.shape, y_features.shape)
        input = (x_features* y_features).sum(dim=-1, keepdim=True)  # b x f x 1 Tensor
        return input
    

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layer_num = num_layers
        for i in range(self.layer_num):
            if self.layer_num == 1:
                lin = nn.Linear(input_size, output_size)
                self.layers.append(lin)
            elif i == 0:
                lin = nn.Linear(input_size, hidden_size)
                self.layers.append(lin)
            elif i == self.layer_num-1:
                lin = nn.Linear(hidden_size, output_size)
                self.layers.append(lin)
            else:
                lin = nn.Linear(hidden_size, hidden_size)
                self.layers.append(lin)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


    
class CustomPINN_Green2D_2(nn.Module):
    def __init__(self, dims: int, output_size: int, hidden_size: int, num_layers: int,  domain: tuple, l_weights: bool):        
        super(CustomPINN_Green2D_2, self).__init__()
        self.dims = dims
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.domain = domain
        self.area = (domain[3]-domain[2])*(domain[1]-domain[0])
        self.phi = PINN_NN_2(input_size=dims, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)
        self.psi = PINN_NN_2(input_size=dims, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)
        self.l_weights = l_weights
        self.quadrature_weights = MLP(input_size=2, hidden_size=32, output_size=1, num_layers=num_layers)

    def forward(self, x, y):
        '''
        x is the input coordinate for u(x) = int (G(x,y) * f(y) dy).
        y is the parameter along which we integrate.
        x: b x f x 2 Tensor; b - batch size of coordinates, f - size of f_mesh, 2 - 2D
        y: b x f x 2 Tensor; b - batch size of coordinates, f - size of f_mesh, 2 - 2D
        '''
        phi = self.phi(x,y)
        psi = self.psi(x,y)
        log_term = torch.log((torch.sqrt(((x-y)**2).sum(-1)))+ 1e-8).view(phi.shape)
        val = (phi * log_term + psi)
        if self.l_weights == True:
            weight = (self.quadrature_weights(y)**2) / self.area
            val *= weight
            
        return val[...,0]
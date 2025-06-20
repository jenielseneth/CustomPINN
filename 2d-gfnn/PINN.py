
import torch.nn as nn
import torch

class PINN_NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(PINN_NN, self).__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.layer_num = num_layers
        for i in range(self.layer_num):
            if self.layer_num == 1:
                lin = nn.Linear(input_size, output_size)
                self.layers.append(lin)
            elif i == 0:
                lin = nn.Linear(input_size, hidden_size)
                self.layers.append(lin)
                norm = nn.LayerNorm(hidden_size)
                self.norms.append(norm)
            elif i == self.layer_num-1:
                lin = nn.Linear(hidden_size, output_size)
                self.layers.append(lin)
            else:
                lin = nn.Linear(hidden_size, hidden_size)
                self.layers.append(lin) 
                norm = nn.LayerNorm(hidden_size)
                self.norms.append(norm)
            self.dropout.append(nn.Dropout(p=0.1))

    def forward(self, x, y):
        input = torch.cat((x,y), dim=-1)
        for i, layer in enumerate(self.layers):
            input = layer(input)
            if i < len(self.layers) - 1:
                input = torch.relu(input)
            input = self.dropout[i](input)
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


    
class CustomPINN_Green2D(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int, domain: tuple, l_weights: bool):        
        super(CustomPINN_Green2D, self).__init__()
        self.hidden_size = hidden_size
        self.domain = domain
        self.area = (domain[3]-domain[2])*(domain[1]-domain[0])
        self.phi = PINN_NN(input_size=4, hidden_size=hidden_size, output_size=1, num_layers=num_layers)
        self.psi = PINN_NN(input_size=4, hidden_size=hidden_size, output_size=1, num_layers=num_layers)
        self.l_weights = l_weights
        self.quadrature_weights = MLP(input_size=2, hidden_size=32, output_size=1, num_layers=num_layers)

    def forward(self, x, y):
        '''
        x is the input coordinate for u(x) = int (G(x,y) * f(y) dy). \n
        y is the parameter along which we integrate. \n
        :param Tensor x: b x f x 2 Tensor; b - batch size of coordinates, f - size of f_mesh, 2 - 2D
        :param Tensor y: b x f x 2 Tensor; b - batch size of coordinates, f - size of f_mesh, 2 - 2D
        :return: b x f Tensor; b - batch size of coordinates, f - size of f_mesh
        :rtype: Tensor
        '''
        assert len(x.shape) == 3 and len(y.shape) == 3, "x and y must be 3D tensors with shape (batch_size, f_size, 2)"
        phi = self.phi(x,y)
        psi = self.psi(x,y)

        log_term = torch.log((torch.sqrt(((x-y)**2).sum(-1)))).view(phi.shape)
        val = (phi * log_term + psi)
        if self.l_weights == True:
            weight = (self.quadrature_weights(y)**2) / self.area
            val *= weight
                        
        return val[...,0]
        

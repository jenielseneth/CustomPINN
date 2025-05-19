
import torch.nn as nn
import torch

class PINN_NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PINN_NN, self).__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.layer_num = 5
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

    
class CustomPINN_Green2D(nn.Module):
    def __init__(self, dims: int, output_size: int, hidden_size: int):        
        super(CustomPINN_Green2D, self).__init__()
        self.dims = dims
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.phi = PINN_NN(input_size=dims, hidden_size=hidden_size, output_size=output_size)
        self.psi = PINN_NN(input_size=dims, hidden_size=hidden_size, output_size=output_size)

    def forward(self, x, y):
        phi = self.phi(x,y)
        psi = self.psi(x,y)
        log_term = torch.log(torch.abs(x-y).sum(-1)).view(phi.shape)
        val = phi * log_term + psi
        if self.output_size == 1:
            return val[:,0]
        else:
            return val

###Emilio Recommendation
# class MLP(nn.Module):
#     def __init__(self, input_size, output_size, hidden_size):
#         super(MLP, self).__init__()
#         self.layers = nn.ModuleList()
#         self.norms = nn.ModuleList()
#         self.dropout = nn.ModuleList()
#         self.layer_num = 5
#         for i in range(self.layer_num):
#             if self.layer_num == 1:
#                 lin = nn.Linear(input_size, output_size)
#                 self.layers.append(lin)
#             elif i == 0:
#                 lin = nn.Linear(input_size, hidden_size)
#                 self.layers.append(lin)
#                 norm = nn.LayerNorm(hidden_size)
#                 self.norms.append(norm)
#             elif i == self.layer_num-1:
#                 lin = nn.Linear(hidden_size, output_size)
#                 self.layers.append(lin)
#             else:
#                 lin = nn.Linear(hidden_size, hidden_size)
#                 self.layers.append(lin)
#                 norm = nn.LayerNorm(hidden_size)
#                 self.norms.append(norm)
#             self.dropout.append(nn.Dropout(p=0.1))

#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = layer(x)
#             if i < len(self.layers) - 1:
#                 x = torch.relu(x)
#             x = self.dropout[i](x)
#         return x

# class CustomPINN_Green2D(nn.Module):
#     def __init__(self, dims: int, output_size: int, hidden_size: int):        
#         super(CustomPINN_Green2D, self).__init__()
#         self.dims = dims
#         self.output_size = output_size
#         self.hidden_size = hidden_size
#         self.lifting_operator = MLP(input_size=dims, hidden_size=hidden_size, output_size=hidden_size)
#         self.phi = MLP(input_size=hidden_size, hidden_size=hidden_size, output_size=hidden_size)
#         self.psi = MLP(input_size=hidden_size, hidden_size=hidden_size, output_size=hidden_size)

#     def forward(self, x, y):
#         '''
#         x: b x 2 Tensor
#         y: b x 2 Tensor
#         '''
#         lifted_x = self.lifting_operator(x)
#         lifted_y = self.lifting_operator(y)
#         phi = (self.phi(lifted_x) * self.phi(lifted_y)).sum(-1)
#         psi = (self.psi(lifted_x) * self.psi(lifted_y)).sum(-1)
#         log_term = torch.log(torch.abs(x-y).sum(-1)).view(phi.shape)
#         val = phi * log_term + psi
#         return val

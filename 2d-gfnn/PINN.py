
import math
from typing import Union
import torch.nn as nn
import torch

class PINN_NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(PINN_NN, self).__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.layer_num = num_layers
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size for _ in range(self.layer_num-2)]
        else:
            assert len(hidden_size) == self.layer_num-2
        for i in range(self.layer_num-1):
                
            if self.layer_num == 1:
                lin = nn.Linear(input_size, output_size)
                self.layers.append(lin)
            elif i == 0:
                lin = nn.Linear(input_size, hidden_size[i])
                self.layers.append(lin)
            elif i == self.layer_num-2:
                lin = nn.Linear(hidden_size[i-1], output_size)
                self.layers.append(lin)
            else:
                lin = nn.Linear(hidden_size[i-1],  hidden_size[i])
                self.layers.append(lin) 

    def forward(self, x, s, log_term: torch.Tensor = None):
        # Expand x and s to match the expected input shape
        if log_term is not None:
            assert x.dim() == s.dim() == 3, "When feeding in log terms, the data passed must be of shape b x f x 2"

        if x.dim() == 2:
            assert s.dim() == 2, "s must be a 2D tensor if x is 2D."
            x = x[:, None, :].expand(-1, s.shape[0], -1)  # b x f x 2 Tensor
            s = s[None, :, :].expand(x.shape[0], -1, -1)  # b x f x 2 Tensor
        

        # Temporary test with log
        if log_term is None:
            input = torch.cat((x, s), dim=-1)
        else:
            input = torch.cat((x,s,log_term), dim=-1)


        for i, layer in enumerate(self.layers):
            input = layer(input)
            if i < len(self.layers) - 1:
                input = torch.tanh(input)
            # input = self.dropout[i](input)
        return input
    
class PINN_NN_no_cat(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.layer_num = num_layers
        self.beginning_lift_x = nn.Linear(int(input_size/2), hidden_size)
        self.beginning_lift_s = nn.Linear(int(input_size/2), hidden_size, bias=False)
        input_size = hidden_size
        for i in range(self.layer_num - 1):
            if self.layer_num == 1:
                lin = nn.Linear(input_size, output_size)
                self.layers.append(lin)
            elif i == 0:
                lin = nn.Linear(input_size, hidden_size)
                self.layers.append(lin)
                norm = nn.LayerNorm(hidden_size)
                self.norms.append(norm)
            elif i == self.layer_num-2:
                lin = nn.Linear(hidden_size, output_size)
                self.layers.append(lin)
            else:
                lin = nn.Linear(hidden_size, hidden_size)
                self.layers.append(lin) 
                norm = nn.LayerNorm(hidden_size)
                self.norms.append(norm)
            self.dropout.append(nn.Dropout(p=0.1))

    def forward(self, x, s):
        # Expand x and y to match the expected input shape
        if x.dim() == 2:
            assert s.dim() == 2, "s must be a 2D tensor if x is 2D."
            x = x[:, None, :].expand(-1, s.shape[0], -1)  # b x f x 2 Tensor
            s = s[None, :, :].expand(x.shape[0], -1, -1)  # b x f x 2 Tensor

        # Replace torch.cat with mathematical equivalent:
        #   nn.Linear: output = input Aᵀ + b -> input = [x, s] -> A = [A₁, A₂] -> [x, s] [A₁ᵀ, A₂ᵀ] + b = x A₁ᵀ + s A₂ᵀ + b
        x_lift = self.beginning_lift_x(x)
        s_lift = self.beginning_lift_s(s)
        input = x_lift + s_lift
        input = torch.tanh(input)  # Apply tanh activation to the lifted input

        for i, layer in enumerate(self.layers):
            input = layer(input)
            if i < len(self.layers) - 1:
                input = torch.tanh(input)
            # input = self.dropout[i](input)
        return input
    
class PINN_NN_Dot(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
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

    def forward(self, x, s, log_term: torch.Tensor = None):
        '''
        Parameters:
            x: b x f x h Tensor; b - batch size of coordinates, f - size of f_mesh, h - size of hidden layer
            s: b x f x h Tensor; b - batch size of coordinates, f - size of f_mesh, h - size of hidden layer
            log_term: b x f x 1 Tensor; b - batch size of coordinates, f - size of f_mesh
        '''
        # Expand x and y to match the expected input shape
        if x.dim() == 2:
            assert s.dim() == 2, "s must be a 2D tensor if x is 2D."
            x = x[:, None, :].expand(-1, s.shape[0], -1)  # b x f x 2 Tensor
            s = s[None, :, :].expand(x.shape[0], -1, -1)  # b x f x 2 Tensor

        input = (x*s).sum(-1, keepdim=True)
        if log_term is not None:
            input = torch.cat((input, log_term), dim=-1)

        for i, layer in enumerate(self.layers):
            input = layer(input)
            if i < len(self.layers) - 1:
                input = torch.tanh(input)
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
                x = torch.tanh(x)
        return x


    
class CustomPINN_Green2D(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int):        
        super(CustomPINN_Green2D, self).__init__()
        self.hidden_size = hidden_size
        # self.domain = domain
        # self.area = (domain[3]-domain[2])*(domain[1]-domain[0])
        self.phi = PINN_NN(input_size=4, hidden_size=hidden_size, output_size=1, num_layers=num_layers)
        self.psi = PINN_NN(input_size=4, hidden_size=hidden_size, output_size=1, num_layers=num_layers)
        # self.l_weights = l_weights
        self.quadrature_weights = MLP(input_size=2, hidden_size=32, output_size=1, num_layers=num_layers)

    def forward(self, x, y, area = None):
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
        if area is not None:
            weight = (self.quadrature_weights(y)**2) / self.area
            val *= weight
                        
        return val[...,0]

class CustomPINN_Green2D_Baseline(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int):        
        super(CustomPINN_Green2D_Baseline, self).__init__()
        self.hidden_size = hidden_size
        self.phi = PINN_NN(input_size=4, hidden_size=hidden_size, output_size=1, num_layers=num_layers)
        self.psi = PINN_NN(input_size=4, hidden_size=hidden_size, output_size=1, num_layers=num_layers)
        self.quadrature_weights = MLP(input_size=2, hidden_size=32, output_size=1, num_layers=num_layers)
    def forward(self, x, y, area = None):
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
        val = phi
        if area is not None:
            weight = (self.quadrature_weights(y)**2) / area
            val *= weight
                        
        return val[...,0]
    

class CustomPINN_Green2D_LogBaseline(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int):        
        super(CustomPINN_Green2D_LogBaseline, self).__init__()
        self.hidden_size = hidden_size
        self.phi = PINN_NN(input_size=4, hidden_size=hidden_size, output_size=1, num_layers=num_layers)
        self.psi = PINN_NN(input_size=4, hidden_size=hidden_size, output_size=1, num_layers=num_layers)
        self.quadrature_weights = MLP(input_size=2, hidden_size=32, output_size=1, num_layers=num_layers)
    def forward(self, x, y, area = None):
        '''
        x is the input coordinate for u(x) = int (G(x,y) * f(y) dy). \n
        y is the parameter along which we integrate. \n
        :param Tensor x: b x f x 2 Tensor; b - batch size of coordinates, f - size of f_mesh, 2 - 2D
        :param Tensor y: b x f x 2 Tensor; b - batch size of coordinates, f - size of f_mesh, 2 - 2D
        :return: b x f Tensor; b - batch size of coordinates, f - size of f_mesh
        :rtype: Tensor
        '''
        assert len(x.shape) == 3 and len(y.shape) == 3, "x and y must be 3D tensors with shape (batch_size, f_size, 2)"
        log_term = torch.log((torch.sqrt(((x-y)**2).sum(-1))))[..., None]
        val = log_term
        if area is not None:
            weight = (self.quadrature_weights(y)**2) / area
            val *= weight
                        
        return val[...,0]
    
class CustomPINN_Green2D_PoissonExplicit(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int):        
        super(CustomPINN_Green2D_PoissonExplicit, self).__init__()
        self.hidden_size = hidden_size
        self.lifting_operation = MLP(input_size=2, hidden_size=hidden_size, output_size=hidden_size, num_layers=1)
        self.phi_func = PINN_NN(input_size=hidden_size*2, hidden_size=hidden_size, output_size=1, num_layers=num_layers)
        self.phi = lambda x, s: self.phi_func(self.lifting_operation(x), self.lifting_operation(s))
        self.psi_func = PINN_NN(input_size=hidden_size*2, hidden_size=hidden_size, output_size=1, num_layers=num_layers)
        self.psi = lambda x, s: self.psi_func(self.lifting_operation(x), self.lifting_operation(s))
        self.quadrature_weights = MLP(input_size=2, hidden_size=32, output_size=1, num_layers=num_layers)

    def forward(self, x, s, area = None):
        '''
        x is the input coordinate for u(x) = int (G(x,y) * f(y) dy). \n
        s is the parameter along which we integrate. \n
        :param Tensor x: b x f x 2 Tensor; b - batch size of coordinates, f - size of f_mesh, 2 - 2D
        :param Tensor s: b x f x 2 Tensor; b - batch size of coordinates, f - size of f_mesh, 2 - 2D
        :return: b x f Tensor; b - batch size of coordinates, f - size of f_mesh
        :rtype: Tensor
        '''
        assert len(x.shape) == 3 and len(s.shape) == 3, f"x ({x.shape}) and s ({s.shape}) must be 3D tensors with shape (batch_size, f_size, 2)"
        # phi = self.phi(x,y)
        psi = self.psi(x, s)
        log_term = torch.log((torch.sqrt(((x-s)**2).sum(-1)))).view(psi.shape)
        val = (-1/(2*torch.pi)*log_term + psi)
        # val = psi
        if area is not None:
            weight = (self.quadrature_weights(s)**2) / self.area
            val *= weight
                        
        return val[...,0]

class CustomPINN_Green2D_PoissonExplicit_W_Log(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int):        
        super(CustomPINN_Green2D_PoissonExplicit_W_Log, self).__init__()
        self.hidden_size = hidden_size
        self.lifting_operation = MLP(input_size=2, hidden_size=hidden_size, output_size=hidden_size, num_layers=1)
        self.log_lifting_operation = MLP(input_size=1, hidden_size=hidden_size, output_size=hidden_size, num_layers=1)
        self.phi_func = PINN_NN(input_size=hidden_size*2, hidden_size=hidden_size, output_size=1, num_layers=num_layers)
        self.phi = lambda x, s: self.phi_func(x, s)
        self.psi_func = PINN_NN(input_size=hidden_size*3, hidden_size=hidden_size, output_size=1, num_layers=num_layers) # input_size = hidden_size*3 because self.lifting_operation(x) and self.lifting_operation(s) are both of size hidden_size, and we add the log term
        self.psi = lambda x, s: self.psi_func(self.lifting_operation(x), self.lifting_operation(s), self.log_lifting_operation(torch.log((torch.sqrt(((x-s)**2).sum(-1)))).view(*x.shape[0:2], 1)))
        self.quadrature_weights = MLP(input_size=2, hidden_size=32, output_size=1, num_layers=num_layers)

    def forward(self, x, s, area = None):
        '''
        x is the input coordinate for u(x) = int (G(x,y) * f(y) dy). \n
        y is the parameter along which we integrate. \n
        :param Tensor x: b x f x 2 Tensor; b - batch size of coordinates, f - size of f_mesh, 2 - 2D
        :param Tensor y: b x f x 2 Tensor; b - batch size of coordinates, f - size of f_mesh, 2 - 2D
        :return: b x f Tensor; b - batch size of coordinates, f - size of f_mesh
        :rtype: Tensor
        '''
        assert len(x.shape) == 3 and len(s.shape) == 3, f"x ({x.shape}) and s ({s.shape}) must be 3D tensors with shape (batch_size, f_size, 2)"
        # phi = self.phi(x,y)
        psi = self.psi(x, s)
        log_term = torch.log((torch.sqrt(((x-s)**2).sum(-1)))).view(psi.shape)
        val = (-1/(2*torch.pi)*log_term + psi)
        # val = psi
        if area is not None:
            weight = (self.quadrature_weights(s)**2) / self.area
            val *= weight
                        
        return val[...,0]

class CustomPINN_Green2D_PoissonExplicit_Dot(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int):        
        super().__init__()
        self.hidden_size = hidden_size
        self.lifting_operation = MLP(input_size=2, hidden_size=hidden_size, output_size=hidden_size, num_layers=1)
        self.phi_func = PINN_NN_Dot(input_size=1, hidden_size=hidden_size, output_size=1, num_layers=num_layers)
        self.phi = lambda x, y: self.phi_func(self.lifting_operation(x), self.lifting_operation(y))
        self.psi_func = PINN_NN_Dot(input_size=1, hidden_size=hidden_size, output_size=1, num_layers=num_layers)
        self.psi = lambda x, y: self.psi_func(self.lifting_operation(x), self.lifting_operation(y))
        self.quadrature_weights = MLP(input_size=2, hidden_size=32, output_size=1, num_layers=num_layers)

    def forward(self, x, y, area = None):
        '''
        x is the input coordinate for u(x) = int (G(x,y) * f(y) dy). \n
        y is the parameter along which we integrate. \n
        :param Tensor x: b x f x 2 Tensor; b - batch size of coordinates, f - size of f_mesh, 2 - 2D
        :param Tensor y: b x f x 2 Tensor; b - batch size of coordinates, f - size of f_mesh, 2 - 2D
        :return: b x f Tensor; b - batch size of coordinates, f - size of f_mesh
        :rtype: Tensor
        '''
        assert len(x.shape) == 3 and len(y.shape) == 3, f"x ({x.shape}) and y ({y.shape}) must be 3D tensors with shape (batch_size, f_size, 2)"
        # phi = self.phi(x,y)
        psi = self.psi(x, y)

        log_term = torch.log((torch.sqrt(((x-y)**2).sum(-1)))).view(psi.shape)
        val = ((-1/(2*torch.pi))*log_term + psi)
        if area is not None:
            weight = (self.quadrature_weights(y)**2) / self.area
            val *= weight
                        
        return val[...,0]
    
class CustomPINN_Green2D_PoissonExplicit_Fourier(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int):        
        super().__init__()
        self.hidden_size = hidden_size

        def fourier_feature(x):
            return torch.cat((torch.cos(2*torch.pi*self.gaussian_matrix(x)), 
                              torch.sin(2*torch.pi*self.gaussian_matrix(x))), dim=-1)

        # Intialize Gaussian Matrix
        self.gaussian_matrix = torch.nn.Linear(in_features=2, out_features=hidden_size)
        torch.nn.init.normal_(self.gaussian_matrix.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.gaussian_matrix.bias, mean=0.0, std=0.01)

        self.phi_func = PINN_NN(input_size=hidden_size*4, hidden_size=hidden_size*4, output_size=1, num_layers=num_layers)
        self.phi = lambda x, y: self.phi_func(fourier_feature(x), fourier_feature(y))
        self.psi_func = PINN_NN(input_size=hidden_size*4, hidden_size=hidden_size*4, output_size=1, num_layers=num_layers)
        self.psi = lambda x, y: self.psi_func(fourier_feature(x), fourier_feature(y))
        self.quadrature_weights = MLP(input_size=2, hidden_size=32, output_size=1, num_layers=num_layers)

    def forward(self, x, y, area = None):
        '''
        x is the input coordinate for u(x) = int (G(x,y) * f(y) dy). \n
        y is the parameter along which we integrate. \n
        :param Tensor x: b x f x 2 Tensor; b - batch size of coordinates, f - size of f_mesh, 2 - 2D
        :param Tensor y: b x f x 2 Tensor; b - batch size of coordinates, f - size of f_mesh, 2 - 2D
        :return: b x f Tensor; b - batch size of coordinates, f - size of f_mesh
        :rtype: Tensor
        '''
        assert len(x.shape) == 3 and len(y.shape) == 3, f"x ({x.shape}) and y ({y.shape}) must be 3D tensors with shape (batch_size, f_size, 2)"
        # phi = self.phi(x,y)
        psi = self.psi(x, y)
        log_term = torch.log((torch.sqrt(((x-y)**2).sum(-1)))).view(psi.shape)
        val = (log_term + psi)
        if area is not None:
            weight = (self.quadrature_weights(y)**2) / self.area
            val *= weight
                        
        return val[...,0]
    
class CustomPINN_Green2D_PoissonExplicit_Fourier_Dot(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int):        
        super().__init__()
        self.hidden_size = hidden_size

        def fourier_feature(gaussian_matrix, x):
            return torch.cat((torch.cos(2*torch.pi*gaussian_matrix(x)), 
                              torch.sin(2*torch.pi*(gaussian_matrix(x)))), dim=-1)
        
        self.fourier_feature = fourier_feature
        # Intialize Gaussian Matrix
        self.gaussian_matrix = torch.nn.Linear(in_features=2, out_features=hidden_size, bias=False)
        torch.nn.init.normal_(self.gaussian_matrix.weight, mean=0.0, std=0.01)
        self.gaussian_matrix.requires_grad = False
        self.phi_func = PINN_NN_Dot(input_size=1, hidden_size=hidden_size, output_size=1, num_layers=num_layers)
        self.phi = lambda x, s: self.phi_func(fourier_feature(self.gaussian_matrix, x), fourier_feature(self.gaussian_matrix, s))
        self.psi_func = PINN_NN_Dot(input_size=1, hidden_size=hidden_size, output_size=1, num_layers=num_layers)
        self.psi = lambda x, s: self.psi_func(fourier_feature(self.gaussian_matrix, x), fourier_feature(self.gaussian_matrix, s))
        self.quadrature_weights = MLP(input_size=2, hidden_size=32, output_size=1, num_layers=num_layers)

    def forward(self, x, s, area = None):
        '''
        x is the input coordinate for u(x) = int (G(x,y) * f(y) dy). \n
        y is the parameter along which we integrate. \n
        :param Tensor x: b x f x 2 Tensor; b - batch size of coordinates, f - size of f_mesh, 2 - 2D
        :param Tensor y: b x f x 2 Tensor; b - batch size of coordinates, f - size of f_mesh, 2 - 2D
        :return: b x f Tensor; b - batch size of coordinates, f - size of f_mesh
        :rtype: Tensor
        '''
        assert len(x.shape) == 3 and len(s.shape) == 3, f"x ({x.shape}) and y ({s.shape}) must be 3D tensors with shape (batch_size, f_size, 2)"
        # phi = self.phi(x,y)
        psi = self.psi(x, s)
        log_term = torch.log((torch.sqrt(((x-s)**2).sum(-1)))).view(psi.shape)
        val = (log_term + psi)
        if area is not None:
            weight = (self.quadrature_weights(s)**2) / self.area
            val *= weight
                        
        return val[...,0]
    

class CustomPINN_Green2D_Fourier_Dot(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int):        
        super().__init__()
        self.hidden_size = hidden_size

        def fourier_feature(x):
            return torch.cat((torch.cos(2*torch.pi*self.gaussian_matrix(x)), 
                              torch.sin(2*torch.pi*self.gaussian_matrix(x))), dim=-1)

        # Intialize Gaussian Matrix
        # self.gaussian_matrix = torch.normal(mean=0.0, std=0.01, size=(16, 2))
        self.gaussian_matrix = torch.nn.Linear(in_features=2, out_features=hidden_size)
        torch.nn.init.normal_(self.gaussian_matrix.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.gaussian_matrix.bias, mean=0.0, std=0.01)
        self.gaussian_matrix.requires_grad = False

        self.phi_func = PINN_NN_Dot(input_size=1, hidden_size=hidden_size, output_size=1, num_layers=num_layers)
        self.phi = lambda x, y: self.phi_func(fourier_feature(x), fourier_feature(y))
        self.psi_func = PINN_NN_Dot(input_size=1, hidden_size=hidden_size, output_size=1, num_layers=num_layers)
        self.psi = lambda x, y: self.psi_func(fourier_feature(x), fourier_feature(y))
        self.quadrature_weights = MLP(input_size=2, hidden_size=32, output_size=1, num_layers=num_layers)

    def forward(self, x, y, area = None):
        '''
        x is the input coordinate for u(x) = int (G(x,y) * f(y) dy). \n
        y is the parameter along which we integrate. \n
        :param Tensor x: b x f x 2 Tensor; b - batch size of coordinates, f - size of f_mesh, 2 - 2D
        :param Tensor y: b x f x 2 Tensor; b - batch size of coordinates, f - size of f_mesh, 2 - 2D
        :return: b x f Tensor; b - batch size of coordinates, f - size of f_mesh
        :rtype: Tensor
        '''
        assert len(x.shape) == 3 and len(y.shape) == 3, f"x ({x.shape}) and y ({y.shape}) must be 3D tensors with shape (batch_size, f_size, 2)"
        phi = self.phi(x,y)
        psi = self.psi(x,y)
        log_term = torch.log((torch.sqrt(((x-y)**2).sum(-1)))).view(psi.shape)
        val = (phi*log_term + psi)
        if area is not None:
            weight = (self.quadrature_weights(y)**2) / self.area
            val *= weight
                        
        return val[...,0]
    

class Sine(nn.Module):
    """Sine activation function with a learnable frequency parameter."""
    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class SirenChannelMLP(nn.Module):
    """
    A SIREN (Sinusoidal Representation Network) version of a channel-wise MLP.

    This module processes tensors of shape [Batch, Sequence, Channels] using
    1x1 convolutions and sine activation functions, as described in the SIREN paper.
    It includes the specific weight initialization required for SIRENs to function correctly.

    Args:
        input_dim (int): The number of input channels.
        hidden_layers (Union[int, list[int]]): The size of the hidden layers.
        output_dim (int): The number of output channels.
        depth (int): The total number of Conv1d layers. Must be at least 2 for SIREN.
        dropout (float): Dropout probability applied after each hidden layer's activation.
        bias (bool): If True, adds a learnable bias to the convolutions.
        is_first (bool): Must be True if this is the first layer of the entire model
            to apply the special w0=30 and initialization.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_layers: Union[int, list[int]],
        output_dim: int,
        depth: int = 2,
        dropout: float = 0.0,
        bias: bool = True,
        is_first: bool = False
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("SIREN MLP depth must be at least 2.")

        self.depth = depth
        self.input_dim = input_dim
        self.is_first = is_first
        layers_list: list[nn.Module] = []

        # --- 1. Determine layer dimensions ---
        num_hidden_layers = depth - 1
        if isinstance(hidden_layers, int):
            actual_hidden_dims = [hidden_layers] * num_hidden_layers
        else:
            if len(hidden_layers) >= num_hidden_layers:
                actual_hidden_dims = hidden_layers[:num_hidden_layers]
            else:
                # Extend with the last hidden size if not enough are provided
                actual_hidden_dims = hidden_layers + [hidden_layers[-1]] * (num_hidden_layers - len(hidden_layers))

        all_dims = [input_dim] + actual_hidden_dims + [output_dim]

        # --- 2. Build the SIREN network ---
        for i in range(depth):
            in_d, out_d = all_dims[i], all_dims[i+1]
            is_final_layer = (i == depth - 1)

            # Add the linear layer (as a 1x1 convolution)
            layers_list.append(nn.Conv1d(in_d, out_d, kernel_size=1, bias=bias))

            # Add Sine activation to all BUT the final layer
            if not is_final_layer:
                w0 = 30.0 if self.is_first and i == 0 else 1.0
                layers_list.append(Sine(w0=w0))
                if dropout > 0:
                    layers_list.append(nn.Dropout(p=dropout))

        self.network = nn.Sequential(*layers_list)
        self._initialize_weights()

    def _initialize_weights(self):
        """Apply the specific weight initialization required for SIRENs."""
        with torch.no_grad():
            for i, m in enumerate(self.network.modules()):
                if isinstance(m, nn.Conv1d):
                    # Check if this is the first Conv1d layer of the first MLP in the model
                    is_first_conv_in_first_layer = self.is_first and i == 1 # i==1 because module 0 is the Sequential itself

                    if is_first_conv_in_first_layer:
                        # SIREN first layer initialization
                        nn.init.uniform_(m.weight, -1 / self.input_dim, 1 / self.input_dim)
                    else:
                        # SIREN subsequent layer initialization
                        # in_channels is equivalent to fan_in for a 1x1 Conv
                        nn.init.uniform_(m.weight, -math.sqrt(6.0 / m.in_channels), math.sqrt(6.0 / m.in_channels))

                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, s:torch.Tensor) -> torch.Tensor:
        """Forward pass expects a tensor of shape [B, S, C]."""
        # Transpose for Conv1d: [B, S, C_in] -> [B, C_in, S]
        x = torch.cat((x, s), dim=-1)
        x_t = x.transpose(1, 2)
        # Apply the sequential network
        out_conv = self.network(x_t.float())
        # Transpose back: [B, C_out, S] -> [B, S, C_out]
        return out_conv.transpose(1, 2)
    
class SIRENPINN_Explicit(nn.Module):
    def __init__(self, num_layers: int, hidden_size: Union[int, list[int]],):        
        super(SIRENPINN_Explicit, self).__init__()
        self.hidden_size = hidden_size
        # self.domain = domain
        # self.area = (domain[3]-domain[2])*(domain[1]-domain[0])
        self.phi = PINN_NN(input_size=4, hidden_size=hidden_size, output_size=1, num_layers=num_layers)
        self.psi = SirenChannelMLP(input_dim=4, depth=num_layers, hidden_layers=hidden_size, output_dim=1)
        # self.l_weights = l_weights
        self.quadrature_weights = MLP(input_size=2, hidden_size=32, output_size=1, num_layers=num_layers)

    def forward(self, x, y, area = None):
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
        val = (-1/(2*torch.pi)*log_term + psi)
        if area is not None:
            weight = (self.quadrature_weights(y)**2) / self.area
            val *= weight
                        
        return val[...,0]
    
class SIRENPINN(nn.Module):
    def __init__(self, num_layers: int, hidden_size: Union[int, list[int]],):        
        super(SIRENPINN, self).__init__()
        self.hidden_size = hidden_size
        # self.domain = domain
        # self.area = (domain[3]-domain[2])*(domain[1]-domain[0])
        self.phi = SirenChannelMLP(input_dim=4, depth=num_layers, hidden_layers=hidden_size, output_dim=1)
        self.psi = SirenChannelMLP(input_dim=4, depth=num_layers, hidden_layers=hidden_size, output_dim=1)
        # self.l_weights = l_weights
        self.quadrature_weights = MLP(input_size=2, hidden_size=32, output_size=1, num_layers=num_layers)

    def forward(self, x, y, area = None):
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
        val = (phi*log_term + psi)
        if area is not None:
            weight = (self.quadrature_weights(y)**2) / self.area
            val *= weight
                        
        return val[...,0]

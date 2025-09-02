
import math
from typing import Union
import torch.nn as nn
import torch
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)
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

    def forward(self, x, s):
        if x.dim() == 2:
            assert s.dim() == 2, "s must be a 2D tensor if x is 2D."
            x = x[:, None, :].expand(-1, s.shape[0], -1)  # b x f x 2 Tensor
            s = s[None, :, :].expand(x.shape[0], -1, -1)  # b x f x 2 Tensor
        

        input = torch.cat((x, s), dim=-1)


        for i, layer in enumerate(self.layers):
            input = layer(input)
            if i < len(self.layers) - 1:
                input = torch.tanh(input)
            # input = self.dropout[i](input)
        return input
    
class PINN_NN_w_Distance(nn.Module):
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

    def forward(self, x, s):
        # Expand x and s to match the expected input shape
        if x.dim() == 2:
            assert s.dim() == 2, "s must be a 2D tensor if x is 2D."
            x = x[:, None, :].expand(-1, s.shape[0], -1)  # b x f x 2 Tensor
            s = s[None, :, :].expand(x.shape[0], -1, -1)  # b x f x 2 Tensor
        
        distance = torch.sqrt(((x-s)**2).sum(-1))[None] # b x f 
        input = torch.cat((x,s, distance), dim=-1)
        logger.info(input.shape)
        assert False


        for i, layer in enumerate(self.layers):
            input = layer(input)
            if i < len(self.layers) - 1:
                input = torch.tanh(input)
            # input = self.dropout[i](input)
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


class GreensFunction2D(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.phi = self.build_phi()
        self.psi = self.build_psi()
        self.log = self.build_log()

    @abstractmethod
    def build_phi(self):
        """Subclasses must return a Callable (phi)."""
        pass

    @abstractmethod
    def build_psi(self):
        """Subclasses must return a Callable (psi)."""
        pass

    def build_log(self):
        """Subclasses must return a Callable (log)."""
        def log_fn(x, s, phi, psi):
            return torch.log((torch.sqrt((torch.sum((x-s)**2, dim=-1, keepdim=True)))))
        return log_fn

    def forward(self, x, s):
        assert len(x.shape) == 3 and len(s.shape) == 3, "x and y must be 3D tensors with shape (batch_size, f_size, 2)"
        phi = self.phi(x,s)
        psi = self.psi(x,s)
        log = self.log(x,s, phi, psi)

        output = (phi * log + psi)
        return output[..., 0]
    
class CustomPINN_Green2D(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int):        
        super(CustomPINN_Green2D, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def build_phi(self):
        self.phi = PINN_NN(input_size=4, hidden_size=self.hidden_size, output_size=1, num_layers=self.num_layers)
        
    def build_psi(self):
        self.psi = PINN_NN(input_size=4, hidden_size=self.hidden_size, output_size=1, num_layers=self.num_layers)
        
class CustomPINN_Green2D_PoissonExplicit(GreensFunction2D):
    def __init__(self, num_layers: int, hidden_size: Union[int, list[int]]):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        super(SIRENPINN, self).__init__()
    
    def build_phi(self):
        return lambda x, s: -1/(2*torch.pi)

    def build_psi(self):
        return PINN_NN(input_size=2, hidden_size=self.hidden_size, output_size=1, num_layers=self.num_layers)


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
    
class SIRENPINN(GreensFunction2D):
    def __init__(self, num_layers: int, hidden_size: Union[int, list[int]]):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        super(SIRENPINN, self).__init__()
    
    def build_phi(self):
        return SirenChannelMLP(input_dim=4, depth=self.num_layers, hidden_layers=self.hidden_size, output_dim=1)

    def build_psi(self):
        return SirenChannelMLP(input_dim=4, depth=self.num_layers, hidden_layers=self.hidden_size, output_dim=1)

class SIRENPINN_Dumb(GreensFunction2D):
    def __init__(self, num_layers: int, hidden_size: Union[int, list[int]]):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        super(SIRENPINN_Dumb, self).__init__()
    
    def build_phi(self):
        return SirenChannelMLP(input_dim=4, depth=self.num_layers, hidden_layers=self.hidden_size, output_dim=1)

    def build_psi(self):
        return SirenChannelMLP(input_dim=4, depth=self.num_layers, hidden_layers=self.hidden_size, output_dim=1)
    
    def build_log(self):
        def dumb_log_fn(x, s, phi, psi):
            return torch.log((torch.nn.Softplus()(phi)+1.)*(torch.sqrt(torch.sum((x-s)**2, dim=-1, keepdim=True))))
        return dumb_log_fn
    
class SIRENPINN_Explicit(GreensFunction2D):
    def __init__(self, num_layers: int, hidden_size: Union[int, list[int]]):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        super(SIRENPINN_Explicit, self).__init__()
    
    def build_phi(self):
        return lambda x, s: -1/(2*torch.pi)

    def build_psi(self):
        return SirenChannelMLP(input_dim=4, depth=self.num_layers, hidden_layers=self.hidden_size, output_dim=1)
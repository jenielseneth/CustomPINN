import torch
import time
from plot_utils import plot_multiple_points, plot_points
from loss import fetch_quadrature_weights
from random_utils import retrieve_dict_from_json
from data_generation_utils import sample_points
from debugging_utils import check_poisson_2d_harmonic_func, plot_greens_function_animation
from PINN import CustomPINN_Green2D_PoissonExplicit_Fourier_Dot
from constants_utils import Hyperparameters
from pde_utils import greens_function_laplacian_2d
from typing import Callable


# domain = (-50,50,-50,50) 
domain = (0,1,0,1)
# eval_points = sample_random_mesh_points(domain, 400)

#-----------------------------------------------------------------------

# eval_points = sample_points(domain=domain, mesh_size=(100,100), mesh_type="uniform")
# gaussian_matrix = torch.normal(0, 0.01, size=(16, 2))
# def fourier_feature(x):
#     return torch.cat((torch.cos(2*torch.pi*(x@gaussian_matrix.mT)), 
#                         torch.sin(2*torch.pi*(x@gaussian_matrix.mT))), dim=-1)
# evaled = fourier_feature(eval_points)
# x = torch.zeros_like(eval_points) + torch.tensor([0.5, 0.5])
# print((fourier_feature(x)*evaled).sum(-1).shape)
# plot_points(eval_points, (fourier_feature(x)*evaled).sum(-1))

#----------------------------
def f(x, s):
    return (x**2).sum(-1)


x = torch.tensor([[2.0, 2.0], [1.0, 1.0]])
s = torch.tensor([[1.0, 1.0]])
# x = x[:, None, :].expand(-1, s.shape[0], -1)
# s = s[None].expand(x.shape[0], -1, -1)
y = x**2 + x 

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Expand x to a 3x3 matrix
y = x.expand(3, 3)  # shape: (3, 3)

# Apply some function
out = y.sum()       # sum over all elements

# Gradient w.r.t x
grad = torch.autograd.grad(out, x)
print(grad)  # tensor([3., 3., 3.])
assert False
# grad = torch.autograd.grad(y, x, create_graph=True)  # dy/dx = 2x -> 4.0
# print(grad)

def greens_function_laplacian_2d(greens_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 x: torch.Tensor, s: torch.Tensor):
    x.requires_grad = True
    s.requires_grad = True
    u = greens_function(x, s)
    grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    print(grad_u)
    lap = 0.0
    for i in range(x.shape[-1]):
        grad2 = torch.autograd.grad(grad_u[..., i], x, grad_outputs=torch.ones_like(grad_u[..., i]), create_graph=True)[0][..., i]
        print("grad2:", grad2)
        lap += grad2
    x.requires_grad = False
    s.requires_grad = False
    return lap

print(greens_function_laplacian_2d(f, x, s))

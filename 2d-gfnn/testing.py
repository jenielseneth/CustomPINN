import torch
import time
from plot_utils import plot_multiple_points, plot_points
from loss import fetch_quadrature_weights
from random_utils import retrieve_dict_from_json
from data_generation_utils import sample_points
from debugging_utils import check_poisson_2d_harmonic_func, plot_greens_function_animation
from PINN import CustomPINN_Green2D_PoissonExplicit_Fourier_Dot
from constants_utils import Hyperparameters


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
def f(x, y):
    return x.pow(2) + y.pow(2) + x*y + 1

x = torch.ones((128,2), requires_grad=True)

def f_jacobian(x, y):
    jacobian = torch.func.jacrev(f, argnums=(0,1))(x, y)
    return (torch.diag(jacobian[0]), torch.diag(jacobian[1]))

def f_jacobian_x(x, y):
    return torch.diag(torch.func.jacrev(f, argnums=0)(x, y))

def f_jacobian_y(x, y):
    return torch.diag(torch.func.jacrev(f, argnums=1)(x, y))

# start = time.time()
# hessian = torch.func.hessian(f, argnums=(0,1))(x[:, 0], x[:, 1])
# end = time.time()
# print(f"Time taken for Hessian calculation: {end - start} seconds")

# start = time.time()
# jacobian = torch.func.jacrev(f, argnums=(0,1))(x[:, 0], x[:, 1])
# end = time.time()
# print(f"Time taken for Jacobian calculation: {end - start} seconds")

start = time.time()
# hessian = (torch.func.jacfwd(f_jacobian , argnums=(0,1))(x[:, 0], x[:, 1]))
# hessian_xx = torch.diag(hessian[0][0])
# hessian_yy = torch.diag(hessian[1][1])
# final_hessian = torch.vstack((hessian_xx, hessian_yy)).mT
#-------
hessian_xx = torch.diag(torch.func.jacfwd(f_jacobian_x, argnums=0)(x[:, 0], x[:, 1]))
hessian_yy = torch.diag(torch.func.jacfwd(f_jacobian_y, argnums=1)(x[:, 0], x[:, 1]))
hessian = torch.vstack((hessian_xx, hessian_yy)).mT
#--------
# hessian_xx = torch.diag(torch.func.jacfwd(lambda x, y: f_jacobian(x,y)[0] , argnums=0)(x[:, 0], x[:, 1]))
# hessian_yy = torch.diag(torch.func.jacfwd(lambda x, y: f_jacobian(x,y)[1] , argnums=1)(x[:, 0], x[:, 1]))
# final_hessian = torch.vstack((hessian_xx, hessian_yy)).mT
end = time.time()
print(f"Time taken for Hessian calculation: {end - start} seconds")
assert False
jacobian = torch.vstack((torch.diag(jacobian[0]), torch.diag(jacobian[1]))).mT
print("Jacobian:\n", jacobian)
print(x)
print(f(x[:, 0], x[:, 1]))
print("Hessian:\n", hessian)
diagonal = torch.diagonal(hessian)
lap = torch.sum(diagonal)  
print(lap)
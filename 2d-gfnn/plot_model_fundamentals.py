import os
import torch
from matplotlib import pyplot as plt
from PINN import CustomPINN_Green2D
from plot_utils import plot_multiple_points
from data_generation_utils import sample_uniform_mesh_points
from pde_utils import bnd_eval_gf_integral
from expr_generation_utils import expr_to_func, func_input_wrapper
from training_utils import MultiBndDatasetWrapper, MultiDatasetWrapper
from random_utils import find_line_with_keyword
import sympy

main_dir = "./res/20250605_1820/"
data_dir = main_dir + "data/test/"
model_dir = main_dir + "models/model_20250605_183157/"
figure_dir = model_dir + "figures/"

model_info_file = open(model_dir + "main_info.txt", "r")
l_weights_line = find_line_with_keyword(file_path=model_dir + "main_info.txt", keyword="Learn Quadrature Weights", index=14)
l_weights = "True" in l_weights_line

domain = (0,1,0,1)
model = CustomPINN_Green2D(4, 1, 32, num_layers=3, domain=domain, l_weights=l_weights)
model.load_state_dict(torch.load(model_dir + "model.pth"))
model.eval()



# Uniform mesh for plotting
uniform_mesh = sample_uniform_mesh_points(domain, num_points=(50,50))

#Get approximated/ground truth (gt) u values on uniform mesh
# u_uniform = eval_u_integral_2(greens_function=model, coordinates=uniform_mesh, f_mesh=f_mesh, f_values=f_values)
# gt_uniform = u_func(uniform_mesh)
weights_uniform = model.quadrature_weights(uniform_mesh)**2
domain_center = torch.tensor(((domain[1]-domain[0])/2, (domain[3]-domain[2])/2))
psi_uniform = model.psi(torch.zeros_like(uniform_mesh) + domain_center, uniform_mesh)
phi_uniform = model.phi(torch.zeros_like(uniform_mesh) + domain_center, uniform_mesh)

log_term = torch.log((torch.abs(uniform_mesh-domain_center).sum(-1))+ 1e-8).view(psi_uniform.shape)
greens_function_uniform = (phi_uniform * log_term + psi_uniform) * weights_uniform


plot_multiple_points(points_list=[uniform_mesh, uniform_mesh, uniform_mesh, uniform_mesh], 
                     values_list=[weights_uniform, psi_uniform, phi_uniform, greens_function_uniform], 
                     title_list=["Quadrature weight predictions", "Psi Predicition on domain center",
                                 "Phi Predicition on domain center", "Greens function at delta function at domain center"], 
                     cmap_list=["plasma", "viridis","viridis", "viridis"],
                     main_title="Fundamental plots",
                     save_dir=figure_dir, save_name="Fundamentals")
import os
import torch
from matplotlib import pyplot as plt
from PINN import CustomPINN_Green2D
from PINN_2 import CustomPINN_Green2D_2
from plot_utils import plot_multiple_points
from data_generation_utils import sample_uniform_mesh_points
from pde_utils import bnd_eval_gf_integral
from expr_generation_utils import expr_to_func, func_input_wrapper
from training_utils import MultiBndDatasetWrapper
from random_utils import find_line_with_keyword
from loss import fetch_quadrature_weights
import sympy

user_input = input("Enter the res folder we retrieve data from: ")
main_dir = "./res/" + user_input + "/"
if not os.path.exists(main_dir):
    raise IsADirectoryError(f'The directory {main_dir} does not exist.')
print(f"Using main directory: {main_dir}")

user_input = input("Enter the model folder we retrieve data from: ")
model_dir = main_dir + "models/" + user_input + "/"
if not os.path.exists(model_dir):
    raise IsADirectoryError(f'The directory {model_dir} does not exist.')
print(f"Using model: {model_dir}")

data_dir = main_dir + "data/"
figure_dir = model_dir + "figures/"

model_info_file = open(model_dir + "main_info.txt", "r")
l_weights_line = find_line_with_keyword(file_path=model_dir + "main_info.txt", keyword="Learn Quadrature Weights", index=14)
l_weights = "True" in l_weights_line

num_layers = int(find_line_with_keyword(file_path=model_dir + "main_info.txt", keyword="Model Num Layers", index=6).split(":")[1].strip())
hidden_layers = int(find_line_with_keyword(file_path=model_dir + "main_info.txt", keyword="Model Hidden Channels", index=5).split(":")[1].strip())

f_mesh_type_line = find_line_with_keyword(file_path=data_dir + "test_info.txt", keyword="f(x) Mesh Type:", index=5)
f_mesh_type = f_mesh_type_line.split(":")[1].strip()



domain = (0,1,0,1)
model = CustomPINN_Green2D(4, 1, hidden_size=hidden_layers, num_layers=num_layers, domain=domain, l_weights=l_weights)
# model = CustomPINN_Green2D_2(2, hidden_layers, hidden_layers, num_layers=num_layers, domain=domain, l_weights=l_weights)
model.load_state_dict(torch.load(model_dir + "model.pth"))
model.eval()



# Uniform mesh for plotting
uniform_mesh = sample_uniform_mesh_points(domain, num_points=(50,50))[None, :, :]

#Get approximated/ground truth (gt) u values on uniform mesh
# u_uniform = eval_u_integral_2(greens_function=model, coordinates=uniform_mesh, f_mesh=f_mesh, f_values=f_values)
# gt_uniform = u_func(uniform_mesh)
if l_weights:
    weights_uniform = model.quadrature_weights(uniform_mesh)**2
else: 
    weights_uniform = fetch_quadrature_weights(domain, num_points=(50,50), f_mesh_type=f_mesh_type)

domain_center = torch.tensor(((domain[1]-domain[0])/2, (domain[3]-domain[2])/2))
domain_edge = torch.tensor((domain[1], domain[3]))
psi_uniform = model.psi(torch.zeros_like(uniform_mesh) + domain_center, uniform_mesh)[0, :, 0]
phi_uniform = model.phi(torch.zeros_like(uniform_mesh) + domain_center, uniform_mesh)[0, :, 0]

log_term = torch.log((torch.abs(uniform_mesh-domain_center).sum(-1))+ 1e-8).view(psi_uniform.shape)
greens_function_uniform = (phi_uniform * log_term + psi_uniform) * weights_uniform

model_term_center = model(torch.zeros_like(uniform_mesh) + domain_center, uniform_mesh)[0]
model_term_edge = model(torch.zeros_like(uniform_mesh) + domain_edge, uniform_mesh)[0]

print(weights_uniform.shape, psi_uniform.shape, phi_uniform.shape, greens_function_uniform.shape, model_term_center.shape)

uniform_mesh = uniform_mesh[0]  # Remove batch dimension for plotting

plot_multiple_points(points_list=[uniform_mesh, uniform_mesh, uniform_mesh, uniform_mesh, uniform_mesh], 
                     values_list=[weights_uniform, psi_uniform, phi_uniform, model_term_center, model_term_edge], 
                     title_list=["Quadrature weight predictions", "Psi Predicition on domain center",
                                 "Phi Predicition on domain center", "Greens function at delta function at domain center", "Greens function at delta function at domain edge"], 
                     cmap_list=["plasma", "viridis","viridis", "viridis", "viridis"],
                     main_title="Fundamental plots",
                     save_dir=figure_dir, save_name="Fundamentals")
import os
import torch
from matplotlib import pyplot as plt
from PINN import CustomPINN_Green2D
from plot_utils import plot_multiple_points
from data_generation_utils import sample_uniform_mesh_points
from pde_utils import eval_u_integral_2
from expr_generation_utils import expr_to_func, func_input_wrapper
from training_utils import MultiBndDatasetWrapper, MultiDatasetWrapper
import sympy

main_dir = "./res/20250529_1414/"
data_dir = main_dir + "data/test/"
model_dir = main_dir + "models/model_20250529_141603/"
figure_dir = model_dir + "figures/"

domain = (0,1,0,1)
model = CustomPINN_Green2D(4, 1, 32, domain)
model.load_state_dict(torch.load(model_dir + "model.pth"))
model.eval()

test_data = MultiBndDatasetWrapper(data_file_path=data_dir, data_file_name="data_test.pt", domain=domain)

##Get data and ground-truth u(x)
i = 4
assert i < len(test_data)
info_file = open(data_dir + str(i) + "/info.txt", "r")
u_func_txt = info_file.readlines()[1].split(":", 1)[1]
u_func = func_input_wrapper(expr_to_func([sympy.sympify(u_func_txt)]))[0]

data = test_data[i]
f_values, f_mesh = data["f_vals"], data["f_mesh"]
col_crd, col_u_vals = data["col_crd"], data["col_u_vals"]

# Uniform mesh for plotting
uniform_mesh = sample_uniform_mesh_points(domain, num_points=(50,50))

#Get approximated/ground truth (gt) u values on uniform mesh
u_uniform = eval_u_integral_2(greens_function=model, coordinates=uniform_mesh, f_mesh=f_mesh, f_values=f_values)
gt_uniform = u_func(uniform_mesh)

plot_multiple_points(points_list=[uniform_mesh, uniform_mesh, uniform_mesh], 
                     values_list=[u_uniform, gt_uniform, torch.nn.functional.mse_loss(u_uniform, gt_uniform, reduction="none")], 
                     title_list=["Prediction on Uniform Mesh", "Ground Truth on Uniform Mesh", "Loss per point", ], 
                     cmap_list=["viridis", "viridis", "plasma"],
                     main_title="Diffusion Term",
                     save_dir=figure_dir, save_name=str(i))

print(torch.nn.functional.mse_loss(u_uniform, gt_uniform))
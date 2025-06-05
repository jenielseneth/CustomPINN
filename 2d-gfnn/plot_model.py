import os
import torch
from matplotlib import pyplot as plt
from PINN import CustomPINN_Green2D
from plot_utils import plot_multiple_points
from data_generation_utils import sample_uniform_mesh_points
from pde_utils import bnd_eval_gf_integral
from expr_generation_utils import expr_to_func, func_input_wrapper
from training_utils import MultiBndDatasetWrapper, MultiDatasetWrapper
from chebyshev_utils import clenshaw_curtis_weights_2d
from random_utils import find_line_with_keyword
import sympy

main_dir = "./res/20250604_1343/"
data_dir = main_dir + "data/test/"
model_dir = main_dir + "models/model_20250604_181238/"
figure_dir = model_dir + "figures/"

model_info_file = open(model_dir + "main_info.txt", "r")
l_weights_line = find_line_with_keyword(file_path=model_dir + "main_info.txt", keyword="Learn Quadrature Weights", index=14)
l_weights = "True" in l_weights_line

domain = (0,1,0,1)
model = CustomPINN_Green2D(4, 1, 32, num_layers=3, domain=domain, l_weights=l_weights)
model.load_state_dict(torch.load(model_dir + "model.pth"))
model.eval()

test_data = MultiBndDatasetWrapper(data_file_path=data_dir, data_file_name="data_test.pt", domain=domain)

##Get data and ground-truth u(x)
i = 2
assert i < len(test_data)
data_info_file = open(data_dir + str(i) + "/info.txt", "r")
data_info_file_lines = data_info_file.readlines()
u_func_txt = data_info_file_lines[1].split(":", 1)[1]
u_func = func_input_wrapper(expr_to_func([sympy.sympify(u_func_txt)]))[0]
f_func_txt = data_info_file_lines[4].split(":", 1)[1]
f_func = func_input_wrapper(expr_to_func([sympy.sympify(f_func_txt)]))[0]

data = test_data[i]
f_values, f_mesh = data["f_vals"], data["f_mesh"]
col_crd, col_u_vals = data["col_crd"], data["col_u_vals"]

# Uniform mesh for plotting
uniform_mesh = sample_uniform_mesh_points(domain, num_points=(50,50))

#Get approximated/ground truth (gt) u values on uniform mesh
if l_weights:
    u_uniform = bnd_eval_gf_integral(greens_function=model, coordinates=uniform_mesh, f_mesh=f_mesh, f_values=f_values)
else:
    area_ratio = (domain[1]-domain[0])*(domain[3]-domain[2])/(4)
    x_num, y_num = (20, 20) #f_mesh_points used
    weights = clenshaw_curtis_weights_2d((x_num-1, y_num-1)) * area_ratio
    u_uniform = bnd_eval_gf_integral(greens_function=model, coordinates=uniform_mesh, f_mesh=f_mesh, f_values=f_values, weights=weights)
gt_uniform = u_func(uniform_mesh)
f_gt_uniform = f_func(uniform_mesh)

plot_multiple_points(points_list=[uniform_mesh, uniform_mesh, uniform_mesh, uniform_mesh], 
                     values_list=[u_uniform, gt_uniform, torch.nn.functional.mse_loss(u_uniform, gt_uniform, reduction="none"), f_gt_uniform], 
                     title_list=["Prediction on Uniform Mesh", "Ground Truth on Uniform Mesh", "Loss per point", "Source Term on Uniform Mesh"], 
                     cmap_list=["viridis", "viridis", "plasma", "viridis"],
                     main_title="Loss Term: " + str(torch.nn.functional.mse_loss(u_uniform, gt_uniform).item()),
                     save_dir=figure_dir, save_name=str(i))

print()
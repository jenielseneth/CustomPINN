import os
import torch
from matplotlib import pyplot as plt
from PINN import CustomPINN_Green2D
from PINN_2 import CustomPINN_Green2D_2
from plot_utils import plot_multiple_points
from data_generation_utils import sample_points, sample_uniform_mesh_points
from pde_utils import bnd_eval_gf_integral
from expr_generation_utils import expr_to_func, func_input_wrapper
from training_utils import MultiBndDatasetWrapper, UpdatedMultiDatasetWrapper, get_collocation_boundary_idx
from chebyshev_utils import cheb_2d_impl_2, clenshaw_curtis_weights_2d
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

domain = (0,1,0,1)
model = CustomPINN_Green2D(4, 1, hidden_layers, num_layers=num_layers, domain=domain, l_weights=l_weights)
# model = CustomPINN_Green2D_2(2, hidden_layers, hidden_layers, num_layers=num_layers, domain=domain, l_weights=l_weights)
model.load_state_dict(torch.load(model_dir + "model.pth"))
model.eval()


test_data = UpdatedMultiDatasetWrapper(data_file_path=data_dir, data_file_name="data_test.pt", domain=domain)

num_data = len(test_data.f_meshes)


f_mesh_quadrature_points_txt = find_line_with_keyword(data_dir+ "train_info.txt", "f(x) Mesh Size:", index=6).split(":")[-1].strip()
f_mesh_quadrature_points = tuple(map(int, f_mesh_quadrature_points_txt.split(",")))
test_mesh_quadrature_points_txt = find_line_with_keyword(data_dir+ "test_info.txt", "u(x) Mesh Size:", index=3).split(":")[-1].strip()
test_mesh_quadrature_points = tuple(map(int, test_mesh_quadrature_points_txt.split(",")))

test_mesh_type_txt = find_line_with_keyword(data_dir+ "test_info.txt", "u(x) Mesh Type:", index=4).split(":")[-1].strip()
test_mesh = sample_points(domain=domain, num_points=test_mesh_quadrature_points, mesh_type=test_mesh_type_txt)

uniform_mesh = sample_uniform_mesh_points(domain, num_points=(50,50))

print("l_weights:", l_weights, "num_layers:", num_layers, "hidden_layers:", hidden_layers, "domain:", domain, ".")
print("f_mesh_quadrature_points:", f_mesh_quadrature_points, "test_mesh_quadrature_points:", test_mesh_quadrature_points, "test_mesh_type_txt:", test_mesh_type_txt)
# f_mesh_type_txt = find_line_with_keyword(data_dir+ "test_info.txt", "f(x) Mesh Type:", index=5).split(":")[-1].strip()
# area_ratio = (domain[1]-domain[0])*(domain[3]-domain[2])/(4)
# weights = fetch_quadrature_weights(domain=domain, num_points=(f_mesh_quadrature_points[0], f_mesh_quadrature_points[1]), f_mesh_type=f_mesh_type_txt) * area_ratio

total_loss = 0
for i in range(num_data):
    u_func_txt = find_line_with_keyword(file_path=data_dir+"test_fncs.txt", keyword=str(i+1), index=i).split(";", 1)[0].split(":", 1)[1]
    u_func = func_input_wrapper(expr_to_func([sympy.sympify(u_func_txt)]))[0]
    f_func_txt = find_line_with_keyword(file_path=data_dir+"test_fncs.txt", keyword=str(i+1), index=i).split(";", 1)[1].split(":", 1)[1]
    f_func = func_input_wrapper(expr_to_func([sympy.sympify(f_func_txt)]))[0]

    f_values, f_mesh = test_data.f_values[i], test_data.f_meshes[i]


    # Uniform mesh for plotting

    #Get approximated/ground truth (gt) u values on uniform mesh
    if l_weights:
        u_uniform = bnd_eval_gf_integral(greens_function=model, coordinates=uniform_mesh, f_mesh=f_mesh, f_values=f_values)
        u_test_mesh = bnd_eval_gf_integral(greens_function=model, coordinates=test_mesh, f_mesh=f_mesh, f_values=f_values)
    else:
        area_ratio = (domain[1]-domain[0])*(domain[3]-domain[2])/(4)
        x_num, y_num = (20, 20) #f_mesh_points used
        weights = clenshaw_curtis_weights_2d((x_num-1, y_num-1)) * area_ratio
        u_uniform = bnd_eval_gf_integral(greens_function=model, coordinates=uniform_mesh, f_mesh=f_mesh, f_values=f_values, weights=weights)
        u_test_mesh = bnd_eval_gf_integral(greens_function=model, coordinates=test_mesh, f_mesh=f_mesh, f_values=f_values, weights=weights)

        #Test interpolation of Chebyshev points
        # chebyshev_mesh = sample_points(domain=domain, num_points=test_mesh_quadrature_points)
        # # col, bnd = get_collocation_boundary_idx(domain, chebyshev_mesh)
        # model_chebyshev = bnd_eval_gf_integral(greens_function=model, coordinates=chebyshev_mesh, f_mesh=f_mesh, f_values=f_values, weights=weights)
        # # model_chebyshev[bnd] = 0
        # u_uniform = cheb_2d_impl_2(uniform_mesh, model_chebyshev, chebyshev_size=test_mesh_quadrature_points, domain=domain)

    gt_uniform = u_func(uniform_mesh)
    f_gt_uniform = f_func(uniform_mesh)
    sub_length = test_data.sub_lengths[i]
    gt_test_mesh = test_data[slice(*sub_length)]["u_vals"]

    total_loss += torch.nn.functional.mse_loss(u_uniform, gt_uniform).item()
    plot_multiple_points(points_list=[uniform_mesh, uniform_mesh, uniform_mesh, uniform_mesh, test_mesh, test_mesh], 
                        values_list=[u_uniform, gt_uniform, torch.nn.functional.mse_loss(u_uniform, gt_uniform, reduction="none"), f_gt_uniform, u_test_mesh, gt_test_mesh], 
                        title_list=["Prediction u(x) on Uniform Mesh", "Ground Truth u(x) on Uniform Mesh", "Loss per point", "Source Term f(x) on Uniform Mesh", "Prediction u(x) on Test Mesh", "Ground Truth u(x) on Test Mesh"], 
                        cmap_list=["viridis", "viridis", "plasma", "viridis", "viridis", "viridis"],
                        main_title="Model prediction - Total Loss Term: " + str(torch.nn.functional.mse_loss(u_uniform, gt_uniform).item()),
                        save_dir=figure_dir, save_name=str(i), show=False)

print(f"Total Loss: {total_loss/num_data}")
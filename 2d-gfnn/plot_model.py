import os
import torch
from matplotlib import pyplot as plt
from PINN import CustomPINN_Green2D
from PINN_2 import CustomPINN_Green2D_2
from plot_utils import plot_multiple_points, plot_points
from data_generation_utils import sample_points
from pde_utils import evaluate_greens_function_integral
from expr_generation_utils import expr_to_func, func_input_wrapper
from dataset_utils import GreenPINNDataset, get_interior_boundary_idx
from chebyshev_utils import cheb_2d_impl_2
from random_utils import find_line_with_keyword, retrieve_dict_from_json
from constants_utils import Hyperparameters
from loss import fetch_quadrature_weights
import sympy

user_input = input("Enter the res folder we retrieve data from: ")
main_dir = "./res/" + user_input + "/"
if not os.path.exists(main_dir):
    raise IsADirectoryError(f'The directory {main_dir} does not exist.')

user_input = input("Enter the model directory we get our model from: ")
model_dir = main_dir + "models/" + user_input + "/"
if not os.path.exists(model_dir):
    raise IsADirectoryError(f'The directory {model_dir} does not exist.')

data_dir = main_dir + "data/"
figure_dir = model_dir + "figures/"

config_dict = retrieve_dict_from_json(model_dir + "config.json")
config = Hyperparameters(**config_dict)

test_data = GreenPINNDataset(data_file_path=data_dir, data_file_name="data_test.pt")
model = CustomPINN_Green2D(num_layers=config.num_layers, hidden_size=config.hidden_channels, domain=test_data.constants.domain, l_weights=config.l_weights)
# model = CustomPINN_Green2D_2(num_layers=num_layers, hidden_size=hidden_layers, domain=domain, l_weights=l_weights)

model.load_state_dict(torch.load(model_dir + "model_best_MSELoss().pth"))
model.eval()

num_data = len(test_data.f_meshes)

test_data_mesh_size = test_data.u_mesh_size
test_data_mesh_type = test_data.u_mesh_type

uniform_mesh = sample_points(test_data.constants.domain, mesh_size=(50,50), mesh_type="uniform")
data_mesh = test_data[slice(*(test_data.sub_lengths[0]))]["crd"]
data_mesh = sample_points(test_data.constants.domain, mesh_size=test_data.u_mesh_size, mesh_type="chebyshev") #temporary solution to get around corner issue: when excluding corners during generation, causes issues when chebyshev interpolation
intr, bnd = get_interior_boundary_idx(test_data.constants.domain, data_mesh) # get interior and boundary point indices on the data mesh

if config.l_weights:
    weights = None
else:
    weights = fetch_quadrature_weights(domain=test_data.constants.domain, integration_mesh_size=test_data.constants.integration_mesh_size, integration_mesh_type=test_data.constants.integration_mesh_type) 

total_loss = 0
for i in range(num_data):
    u_func_txt = find_line_with_keyword(file_path=data_dir+"test_fncs.txt", keyword=str(i+1), index=i).split(";", 1)[0].split(":", 1)[1]
    u_func = func_input_wrapper(expr_to_func([sympy.sympify(u_func_txt)]))[0]
    f_func_txt = find_line_with_keyword(file_path=data_dir+"test_fncs.txt", keyword=str(i+1), index=i).split(";", 1)[1].split(":", 1)[1]
    f_func = func_input_wrapper(expr_to_func([sympy.sympify(f_func_txt)]))[0]

    f_values, f_mesh = test_data.f_values[i], test_data.f_meshes[i]

    #Get approximated u(x) on data mesh
    
    u_pred_data_mesh = evaluate_greens_function_integral(greens_function=model, 
                                        evaluation_mesh=data_mesh,
                                        integration_mesh=f_mesh, integration_mesh_values=f_values, weights=weights, dataset_constants=test_data.constants)
    u_pred_data_mesh[bnd] = 0 # Boundary condition
    u_pred_uniform = cheb_2d_impl_2(eval_points=uniform_mesh, values=u_pred_data_mesh, chebyshev_size=test_data.constants.evaluation_mesh_size, domain=test_data.constants.domain)

    #Plot ground truths
    u_gt_uniform_mesh = u_func(uniform_mesh)
    u_gt_data_mesh = test_data[slice(*(test_data.sub_lengths[i]))]["u_vals"]

    #Plot source term on uniform mesh
    source_term = f_func(uniform_mesh)

    #Parameters
    points_list = [uniform_mesh, uniform_mesh, uniform_mesh, uniform_mesh, data_mesh, data_mesh]
    values_list = [u_pred_uniform, u_gt_uniform_mesh, torch.nn.functional.mse_loss(u_pred_uniform, u_gt_uniform_mesh, reduction="none"), source_term, u_pred_data_mesh, u_gt_data_mesh]
    title_list = ["Prediction u(x) on Uniform Mesh", "Ground Truth u(x) on Uniform Mesh", "Loss per point", "Source Term f(x) on Uniform Mesh", "Prediction u(x) on Test Mesh", "Ground Truth u(x) on Test Mesh"]
    cmap_list = ["viridis", "viridis", "plasma", "viridis", "viridis", "viridis"]
    title = "Model Cheb. Interpolation prediction - Total Loss Term: " + str(torch.nn.functional.mse_loss(u_pred_uniform, u_gt_uniform_mesh).item())

    # plot_multiple_points(points_list=points_list, 
    #                     values_list=values_list, 
    #                     title_list=title_list, 
    #                     cmap_list=cmap_list,
    #                     main_title=title,
    #                     save_dir=figure_dir, save_name=str(i), show=False)
    
    print("largest loss for data point", i, ":", torch.max(torch.nn.functional.mse_loss(u_pred_uniform, u_gt_uniform_mesh, reduction="none")).item())
    print("Maximum value at predicted u(x) on uniform:", torch.max(torch.abs(u_pred_uniform)).item())
    print("Maximum value at gt u(x) on uniform:", torch.max(torch.abs(u_gt_uniform_mesh)).item())
    total_loss += torch.nn.functional.mse_loss(u_pred_uniform, u_gt_uniform_mesh).item()

print(f"Total Loss: {total_loss/num_data}")
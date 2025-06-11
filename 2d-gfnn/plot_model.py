import os
import torch
from matplotlib import pyplot as plt
from PINN import CustomPINN_Green2D
from PINN_2 import CustomPINN_Green2D_2
from plot_utils import plot_multiple_points
from data_generation_utils import sample_uniform_mesh_points
from pde_utils import bnd_eval_gf_integral
from expr_generation_utils import expr_to_func, func_input_wrapper
from training_utils import MultiBndDatasetWrapper, UpdatedMultiDatasetWrapper
from chebyshev_utils import clenshaw_curtis_weights_2d
from random_utils import find_line_with_keyword
import sympy


class GTPINN():
    def __init__(self):       
        pass
    def __call__(self, x, y): 
        '''
        x is the input coordinate for u(x) = int (G(x,y) * f(y) dy).
        y is the parameter along which we integrate.
        x: b x f x 2 Tensor; b - batch size of coordinates, f - size of f_mesh, 2 - 2D
        y: b x f x 2 Tensor; b - batch size of coordinates, f - size of f_mesh, 2 - 2D
        '''
        log_term = torch.log((torch.sqrt(((x-y)**2).sum(-1)))+ 1e-9)
        val = -(1/(2*torch.pi) * log_term)
        return val


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

for i in range(num_data):
    u_func_txt = find_line_with_keyword(file_path=data_dir+"test_fncs.txt", keyword=str(i+1), index=i).split(";", 1)[0].split(":", 1)[1]
    u_func = func_input_wrapper(expr_to_func([sympy.sympify(u_func_txt)]))[0]
    f_func_txt = find_line_with_keyword(file_path=data_dir+"test_fncs.txt", keyword=str(i+1), index=i).split(";", 1)[1].split(":", 1)[1]
    f_func = func_input_wrapper(expr_to_func([sympy.sympify(f_func_txt)]))[0]

    f_values, f_mesh = test_data.f_values[i], test_data.f_meshes[i]


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
                        save_dir=figure_dir, save_name=str(i), show=False)

import os
import torch
from matplotlib import pyplot as plt
from PINN import CustomPINN_Green2D
from constants_utils import Hyperparameters
from dataset_utils import GreenPINNDataset
from training_utils import InferenceUtils
from plot_utils import plot_multiple_points
from data_generation_utils import sample_points
from expr_generation_utils import expr_to_func, func_input_wrapper
from random_utils import find_line_with_keyword, retrieve_dict_from_json
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

config_dict = retrieve_dict_from_json(model_dir + "config.json")
config = Hyperparameters(**config_dict)
test_data = GreenPINNDataset(data_file_path=data_dir, data_file_name="data_test.pt")
training_utils = InferenceUtils(constants=test_data.constants, config=config)

model = config.model_cls(**config.model_params)
model.load_state_dict(torch.load(model_dir + "model_best_MSELoss().pth"))
model.eval()


# Uniform mesh for plotting
uniform_mesh_size = (50, 50)
uniform_mesh = sample_points(test_data.constants.domain, mesh_size=uniform_mesh_size, mesh_type="uniform")[None, :, :]

if config.l_weights:
    weights_uniform = model.quadrature_weights(uniform_mesh)**2
else: 
    weights_uniform = fetch_quadrature_weights(test_data.constants.domain, 
                                               integration_mesh_size=uniform_mesh_size, 
                                               integration_mesh_type=test_data.constants.integration_mesh_type)

domain_center = torch.tensor(((test_data.constants.domain[1]-test_data.constants.domain[0])/2, (test_data.constants.domain[3]-test_data.constants.domain[2])/2))
domain_center_mesh = torch.zeros_like(uniform_mesh) + domain_center
domain_edge = torch.tensor((test_data.constants.domain[1], test_data.constants.domain[3]))
domain_edge_mesh = torch.zeros_like(uniform_mesh) + domain_edge
psi_uniform = model.psi(uniform_mesh, domain_center_mesh)[0, :, 0]
phi_uniform = model.phi(uniform_mesh, domain_center_mesh)[0, :, 0]

log_term = torch.log(torch.sqrt(((domain_center_mesh - uniform_mesh)**2).sum(-1)))[0]
model_term_center = model(domain_center_mesh, uniform_mesh)[0] # Remove batch dimension for plotting
model_term_edge = model(domain_edge_mesh, uniform_mesh)[0] # Remove batch dimension for plotting

print(weights_uniform.shape, psi_uniform.shape, phi_uniform.shape, model_term_center.shape)

uniform_mesh = uniform_mesh[0]  # Remove batch dimension for plotting

plot_multiple_points(points_list=[uniform_mesh, uniform_mesh, uniform_mesh, uniform_mesh, [uniform_mesh], [uniform_mesh]], 
                     values_list=[weights_uniform, psi_uniform, phi_uniform, log_term, [model_term_center], [model_term_edge]], 
                     title_list=["Quadrature weight predictions", "Psi Predicition on domain center",
                                 "Phi Predicition on domain center", "Log Term on domain center", "Greens function evaluated at domain center", "Greens function evaluated at domain edge"], 
                     cmap_list=["plasma", "viridis","viridis", "plasma", ["viridis"],[ "viridis"]],
                     main_title="Fundamental plots",
                     axs_size=(3,2),
                     save_dir=figure_dir, save_name="Fundamentals")
import torch
from matplotlib import pyplot as plt
from PINN import CustomPINN_Green2D
from plot_utils import plot_multiple_points
from data_generation_utils import sample_uniform_mesh_points
from pde_utils import evaluate_model
from training_utils import WholeDatasetWrapper
from generate_data import source_term,explicit_u_func_1

dir = "./data/manu_sol_1/"
domain = (0,1,0,1)
model = CustomPINN_Green2D(4, 1, 32)
model.load_state_dict(torch.load(dir + "model.pth"))
model.eval()

data_type = "test"
test_data = WholeDatasetWrapper(collocation_file_path=dir + "collocation_"+ data_type + ".pt", boundary_file_path=dir + "boundary_"+ data_type + ".pt")
coordinates,values = test_data[0:]
filter = torch.where(values != torch.inf)[0]
coordinates = coordinates[filter]
values = values[filter]
uniform_mesh = sample_uniform_mesh_points(domain, num_points=(33,33))


# evaluate_u = get_u_evaluation_func(model, source_term=test_source_term)
u_pred = evaluate_model(model=model, coordinates=coordinates, source_term=source_term, domain=domain, chebyshev=True)
u_uniform = evaluate_model(model=model, coordinates=uniform_mesh, source_term=source_term, domain=domain, chebyshev=True)
gt_uniform = explicit_u_func_1(uniform_mesh)
plot_multiple_points(points_list=[uniform_mesh, uniform_mesh, coordinates, coordinates, coordinates], values_list=[u_uniform, gt_uniform,u_pred, values, torch.nn.functional.mse_loss(u_pred, values, reduction="none")], title_list=["Prediction on Uniform Mesh", "Ground Truth on Uniform Mesh", "Predicted Values", "Ground Truth", "Loss per point", ], cmap_list=["viridis", "viridis", "viridis", "viridis", "plasma"])

print(torch.nn.functional.mse_loss(u_pred, values))
import torch
from matplotlib import pyplot as plt
from PINN import CustomPINN_Green2D
from plot_utils import plot_multiple_points
from data_generation_utils import sample_uniform_mesh_points
from pde_utils import evaluate_model
from training_utils import MultiDatasetWrapper, DatasetWrapper
from generate_data import source_term,explicit_u_func_1

model_dir = "./models/"
domain = (0,1,0,1)
model = CustomPINN_Green2D(4, 1, 32)
model.load_state_dict(torch.load(model_dir + "model.pth"))
model.eval()

data_type = "test"

def u_func(points):
    x, y = points[:,0], points[:,1]
    return 2.47658372789608*x**3*y**3 + 4.42910337153783*torch.sin(2*x)*torch.cos(2*y)

test_data = MultiDatasetWrapper(data_file_path="./data/sin(x*y)/data", data_file_name="data_test.pt")
test_f_values = test_data.f_values
test_f_meshes = test_data.f_meshes
ind_dataset_starts = test_data.start_inds
i = 0
coordinates, u_values, f_inds = test_data[ind_dataset_starts[i]:ind_dataset_starts[i+1]]

filter = torch.where(u_values != torch.inf)[0]
coordinates = coordinates[filter]
u_values = u_values[filter]
uniform_mesh = sample_uniform_mesh_points(domain, num_points=(50,50))


# evaluate_u = get_u_evaluation_func(model, source_term=test_source_term)
u_pred = evaluate_model(model=model, coordinates=coordinates, f_values=test_f_values, f_meshes=test_f_meshes, f_inds=f_inds, domain=domain)
u_uniform = evaluate_model(model=model,coordinates=uniform_mesh,f_values=test_f_values, f_meshes=test_f_meshes, f_inds=len(uniform_mesh)*[i], domain=domain)
gt_uniform = u_func(uniform_mesh)
plot_multiple_points(points_list=[uniform_mesh, uniform_mesh, coordinates, coordinates, uniform_mesh], 
                     values_list=[u_uniform, gt_uniform,u_pred, u_values, torch.nn.functional.mse_loss(u_uniform, gt_uniform, reduction="none")], 
                     title_list=["Prediction on Uniform Mesh", "Ground Truth on Uniform Mesh", "Predicted Values", "Ground Truth", "Loss per point", ], 
                     cmap_list=["viridis", "viridis", "viridis", "viridis", "plasma"],
                     main_title="Diffusion Term sin(x*y)")

print(torch.nn.functional.mse_loss(u_pred, u_values))
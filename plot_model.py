import torch
from matplotlib import pyplot as plt
from PINN import CustomPINN_Green2D
from chebyshev import plot_multiple_points, plot_points
from pde_utils import evaluate_model, get_u_evaluation_func, test_source_term   
from data_generation_utils import generate_points

dir = "./data/manu_sol_1/"
domain = (0,1,0,1)
model = CustomPINN_Green2D(4, 1, 32)
model.load_state_dict(torch.load(dir + "model.pth"))
model.eval()
points = torch.load(dir + "uvalues_test.pt")
coordinates = points["coordinates"]
values = points["values"]
filter = torch.where(values != torch.inf)[0]
coordinates = coordinates[filter]
values = values[filter]

# evaluate_u = get_u_evaluation_func(model, source_term=test_source_term)
u_pred = evaluate_model(model=model, coordinates=coordinates, source_term=test_source_term, domain=domain).detach()

plot_multiple_points(points_list=[coordinates, coordinates, coordinates], values_list=[u_pred, values, torch.nn.functional.mse_loss(u_pred, values, reduction="none")], title_list=["Predicted Values", "Ground Truth", "Loss per point"], cmap_list=["viridis", "viridis", "plasma"])
# print(values, u_pred)
print(torch.nn.functional.mse_loss(u_pred, values)/len(u_pred))
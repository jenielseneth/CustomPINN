import torch
from matplotlib import pyplot as plt
from PINN import CustomPINN_Green2D
from chebyshev import plot_multiple_points, plot_points
from pde_utils import evaluate_model, get_u_evaluation_func
from data_generation import generate_points
model = CustomPINN_Green2D(4, 1, 500)
model.load_state_dict(torch.load(   "model.pth"))
model.eval()
points = torch.load("./data/chebyshev_uvalues_w_bc.pt")
coordinates = points["coordinates"]
values = points["values"]
filter = torch.where(values != torch.inf)[0]
coordinates = coordinates[filter]
values = values[filter]

evaluate_u = get_u_evaluation_func(model)
u_pred = evaluate_model(model=model, coordinates=coordinates, domain=[0,1,0,1]).detach()

plot_multiple_points(points_list=[coordinates, coordinates, coordinates], values_list=[u_pred, values, torch.nn.functional.mse_loss(u_pred, values, reduction="none")], title_list=["Predicted Values", "Ground Truth", "Loss per point"], cmap_list=["viridis", "viridis", "plasma"])
# print(values, u_pred)
print(torch.nn.functional.mse_loss(u_pred, values))
import torch
from matplotlib import pyplot as plt
from PINN import CustomPINN_Green2D
from pde_utils import evaluate_model, get_u_evaluation_func
from data_generation import generate_points
model = CustomPINN_Green2D(4, 1, 32)
model.load_state_dict(torch.load("model.pth"))
model.eval()
points = torch.load("uvalues_test.pt")
coordinates = points["coordinates"]
values = points["values"]

evaluate_u = get_u_evaluation_func(model)
u_pred = evaluate_model(model=model, coordinates=coordinates, domain=[0,1,0,1]).detach()
# generate_points(0, 1, 0, 1, 400, evaluate_u=evaluate_u, file_name="model_plot.pt")
# data = torch.load("model_plot.pt")
# new_coord = data["coordinates"]
# u_pred = data["values"].detach().numpy()
plt.scatter(coordinates[:,0], coordinates[:,1], c=values)
plt.show()
plt.scatter(coordinates[:,0], coordinates[:,1], c=u_pred)
plt.show()
print(values, u_pred)
print(torch.nn.functional.mse_loss(u_pred, values))
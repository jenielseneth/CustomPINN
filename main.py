####Get Dataset for training analytically 
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from training_utils import Dataset, train, test, TrainingDataset, TestDataset
from PINN import CustomPINN_Green2D
from data_generation_utils import generate_points
from loss import CustomDataPredLoss, CustomLoss
from pde_utils import get_u_evaluation_func, test_source_term, greens_function_poisson_eq_2d
##2D example


def source_term(x,y):
    return -6

if __name__ == "__main__":
    x_min, x_max = 0, 1 
    y_min, y_max = 0, 1
    domain = (x_min, x_max, y_min, y_max)
    dir = "./data/manu_sol_1/"
    training_data=Dataset(dir + "uvalues_train.pt")
    trainloader = DataLoader(Dataset(dir + "uvalues_train.pt"), batch_size=256, shuffle=True)
    testloader = DataLoader(Dataset(dir + "uvalues_test.pt"), batch_size=256, shuffle=True)

    model = CustomPINN_Green2D(4, 1, 32)
    f_source_term = source_term

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    loss_fn = CustomLoss(num_collocation_points=200, num_boundary_points=50)

    num_epochs = 300
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(model=model, optimizer=optimizer, dataloader=trainloader, loss_fn=loss_fn, scheduler=scheduler, f_source_term=f_source_term, domain=domain)
        test(model=model, dataloader=testloader, loss_fn=loss_fn, f_source_term=f_source_term, domain=domain)
    
    torch.save(model.state_dict(), dir +  "model.pth")




####Get Dataset for training analytically 
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from training_utils import train, test, TrainingDataset, TestDataset
from PINN import CustomPINN_Green2D
# from data_generation import generate_points, sample_mesh_points
from loss import CustomLoss
from pde_utils import test_source_term
##2D example

if __name__ == "__main__":
    x_min, x_max = 0, 1 
    y_min, y_max = 0, 1
    domain = (x_min, x_max, y_min, y_max)
    
    # train_name = "uvalues_train.pt"
    # test_name = "uvalues_test.pt"
    # generate_points(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, num_points=400, file_name=train_name)
    # generate_points(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, num_points=400, file_name=test_name)
    
    trainloader = DataLoader(TrainingDataset(), batch_size=256, shuffle=True)
    testloader = DataLoader(TestDataset(), batch_size=256, shuffle=True)

    model = CustomPINN_Green2D(4, 1, 32)
    f_source_term = test_source_term

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    loss_fn = CustomLoss()

    num_epochs = 300
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(model=model, optimizer=optimizer, dataloader=testloader, loss_fn=loss_fn, scheduler=scheduler, f_source_term=f_source_term, domain=domain)
        # test(model=model, dataloader=trainloader, loss_fn=loss_fn, f_source_term=f_source_term, domain=domain)
    
    torch.save(model.state_dict(), "model.pth")




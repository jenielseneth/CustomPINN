####Get Dataset for training analytically 
import os
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from training_utils import train, test, MultiDatasetWrapper
from PINN import CustomPINN_Green2D
from loss import CustomDataPredLoss
from generate_data import source_term
##2D example


if __name__ == "__main__":
    x_min, x_max = 0, 1 
    y_min, y_max = 0, 1
    domain = (x_min, x_max, y_min, y_max)
    dir = "./models/"
    if not os.path.exists(dir):
        raise IsADirectoryError("The directory doesn't exist.")
    train_data = MultiDatasetWrapper(data_file_path="./data/sin(x*y)", data_file_name="data_train.pt")
    train_f_values = train_data.f_values
    train_f_meshes = train_data.f_meshes
    test_data = MultiDatasetWrapper(data_file_path="./data/sin(x*y)", data_file_name="data_test.pt")
    test_f_values = test_data.f_values
    test_f_meshes = test_data.f_meshes
    trainloader = DataLoader(train_data, batch_size=128, shuffle=True)
    testloader = DataLoader(test_data, batch_size=128, shuffle=True)

    model = CustomPINN_Green2D(4, 1, 32)
    f_source_term = source_term

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    loss_fn = CustomDataPredLoss(num_eval_points=20)

    num_epochs = 300
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(model=model, optimizer=optimizer, dataloader=trainloader, loss_fn=loss_fn, scheduler=scheduler, f_values=train_f_values, f_meshes=train_f_meshes, domain=domain)
        test(model=model, dataloader=testloader, loss_fn=loss_fn, f_values=test_f_values, f_meshes=test_f_meshes, domain=domain)
    
    torch.save(model.state_dict(), dir +  "model.pth")
    print("Training complete. Saved model to " + dir + "model.pth.")




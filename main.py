####Get Dataset for training analytically 
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from training_utils import BoundaryDataloaderWrapper, DatasetWrapper, WholeDatasetWrapper, train, test
from PINN import CustomPINN_Green2D
from data_generation_utils import generate_points
from loss import CustomDataPredLoss
from pde_utils import get_u_evaluation_func, test_source_term, greens_function_poisson_eq_2d
##2D example


def source_term(x,y):
    return -6

if __name__ == "__main__":
    x_min, x_max = 0, 1 
    y_min, y_max = 0, 1
    domain = (x_min, x_max, y_min, y_max)
    dir = "./data/manu_sol_1/"
    train_data = WholeDatasetWrapper(colocation_file_path=dir + "collocation_train.pt", boundary_file_path=dir + "boundary_train.pt")
    test_data = WholeDatasetWrapper(colocation_file_path=dir + "collocation_test.pt", boundary_file_path=dir + "boundary_test.pt")
    trainloader = DataLoader(train_data, batch_size=128, shuffle=True)
    testloader = DataLoader(test_data, batch_size=128, shuffle=True)

    model = CustomPINN_Green2D(4, 1, 32)
    f_source_term = source_term

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    loss_fn = CustomDataPredLoss(num_eval_points=200)

    num_epochs = 300
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(model=model, optimizer=optimizer, dataloader=trainloader, loss_fn=loss_fn, scheduler=scheduler, f_source_term=f_source_term, domain=domain)
        test(model=model, dataloader=testloader, loss_fn=loss_fn, f_source_term=f_source_term, domain=domain)
    
    torch.save(model.state_dict(), dir +  "model.pth")
    print("Training complete. Saved model to " + dir + "model.pth.")




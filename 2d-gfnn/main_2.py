####Get Dataset for training analytically 
import os
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from training_utils import UpdatedMultiDatasetWrapper, test, train, train_w_bnd_loss, test_w_bnd_loss
from PINN import CustomPINN_Green2D
from PINN_2 import CustomPINN_Green2D_2
from loss import BndDataPredLoss, DataPredLoss
from datetime import datetime
import wandb


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    x_min, x_max = 0, 1 
    y_min, y_max = 0, 1
    domain = (x_min, x_max, y_min, y_max)
    # main_dir = "./res/20250606_1134/"
    # if not os.path.exists(main_dir):
    #     raise IsADirectoryError("The directory doesn't exist.")

    user_input = input("Enter the res folder we retrieve data from: ")
    main_dir = "./res/" + user_input + "/"
    if not os.path.exists(main_dir):
        raise IsADirectoryError(f'The directory {main_dir} does not exist.')
    

    model_dir = main_dir + f"models/model_{timestamp}/" 
    data_dir = main_dir + "data/"


    train_data = UpdatedMultiDatasetWrapper(data_file_path=data_dir, data_file_name="data_train.pt", domain=domain)
    test_data = UpdatedMultiDatasetWrapper(data_file_path=data_dir, data_file_name="data_test.pt", domain=domain)
    training_bs = 128
    test_bs = 128
    trainloader = DataLoader(train_data, batch_size=training_bs, shuffle=True)
    testloader = DataLoader(test_data, batch_size=test_bs, shuffle=True)

    hidden_channels = 64
    num_layers = 5
    learn_quadrature_weights = False # Bool to determine whether to learn quadrature weights or use precomputed.
    model = CustomPINN_Green2D(4, 1, hidden_size=hidden_channels, num_layers=num_layers, domain=domain, l_weights=learn_quadrature_weights)
    # model = CustomPINN_Green2D_2(2, hidden_channels, hidden_size=hidden_channels, num_layers=num_layers, domain=domain, l_weights=learn_quadrature_weights)
    f_mesh_quadrature_points = (20,20)
    loss_fn = DataPredLoss(domain=domain, num_points=f_mesh_quadrature_points, l_weights=learn_quadrature_weights, f_mesh_type=train_data.f_mesh_type)

    lr = 1e-2
    weight_decay = 1e-4
    step_size = 30
    gamma=0.5
    num_epochs = 15
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)




    log_wandb = input("Log to Weights and Biases? (y/n): ").strip().lower() == 'y'
    if log_wandb:
    ## Initialize Weights and Biases
    # Start a new wandb run to track this script.
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="jens1225-eth-zrich",
            # Set the wandb project where this run will be logged.
            project="CustomPINN-Green2D-2",
            config={                         # Track hyperparameters and metadata
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "step_size": step_size,
            "gamma": gamma,
            "num_epochs": num_epochs,
            "hidden_channels": hidden_channels,
            "num_layers": num_layers,
            "training_batch_size": training_bs,
            "test_batch_size": test_bs,
            "domain": domain,
            "learn_quadrature_weights": learn_quadrature_weights,
        },
        
    )
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        total_train_loss = train(model=model, optimizer=optimizer, dataloader=trainloader,
                loss_fn=loss_fn, scheduler=scheduler, domain=domain)
        total_test_loss = test(model=model, dataloader=testloader, loss_fn=loss_fn, domain=domain)
        if log_wandb:
            run.log({"train/loss": total_train_loss, "test/loss": total_test_loss, "epoch": epoch+1})
            
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
        print("Warning: " + model_dir + " already exists.")

    log_file_name = model_dir + "main_info.txt"
    with open(log_file_name, "w") as f: 
        f.write('Domain: ' + ', '.join(map(str,domain)) + "\n")
        f.write('Train Batch Size: ' + str(training_bs) + "\n")
        f.write('Test Batch Size: ' + str(test_bs) + "\n")
        f.write('Length of Training Data: ' + str(len(train_data)) + "\n")
        f.write('Length of Test Data: ' + str(len(test_data)) + "\n")
        f.write('Model Hidden Channels: ' + str(hidden_channels) + "\n")
        f.write('Model Num Layers: ' + str(num_layers) + "\n")
        f.write('Number of Training Epochs: ' + str(num_epochs) + "\n")
        f.write('Optimizer Learning Rate: ' + str(lr) + "\n")
        f.write('Optimizer Weight Decay: ' + str(weight_decay) + "\n")
        f.write('Scheduler Step Size: ' + str(step_size) + "\n")
        f.write('Scheduler Gamma: ' + str(gamma) + "\n")
        f.write('Final Total Train Loss: ' + str(total_train_loss) + "\n")
        f.write('Final Total Test Loss: ' + str(total_test_loss) + "\n")
        f.write('Learn Quadrature Weights: ' + str(learn_quadrature_weights) + "\n")
    
    torch.save(model.state_dict(), model_dir +  "model.pth")
    print("Training complete. Saved model to " + model_dir + "model.pth.")

    if log_wandb:
        run.finish()



####Get Dataset for training analytically 
import os
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from training_utils3d import train, test, MultiDatasetWrapper
from PINN3d import CustomPINN_Green3D
from loss3d import CustomDataPredLoss3d
from datetime import datetime


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    x_min, x_max = 0, 1 
    y_min, y_max = 0, 1
    z_min, z_max = 0, 1
    domain = (x_min, x_max, y_min, y_max, z_min, z_max)
    main_dir = "./res/-1/"
    model_dir = main_dir + f"models/model_{timestamp}/" 
    if not os.path.exists(main_dir):
        raise IsADirectoryError("The directory doesn't exist.")
    
    train_data = MultiDatasetWrapper(data_file_path=main_dir + "data", data_file_name="data_train.pt")
    train_f_values = train_data.f_values
    train_f_meshes = train_data.f_meshes
    test_data = MultiDatasetWrapper(data_file_path=main_dir + "data", data_file_name="data_test.pt")
    test_f_values = test_data.f_values
    test_f_meshes = test_data.f_meshes
    training_bs = 128
    test_bs = 128
    trainloader = DataLoader(train_data, batch_size=training_bs, shuffle=True)
    testloader = DataLoader(test_data, batch_size=test_bs, shuffle=True)

    hidden_channels = 32
    model = CustomPINN_Green3D(6, 1, hidden_size=hidden_channels)

    lr = 1e-2
    weight_decay = 1e-4
    step_size=100
    gamma=0.5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    loss_fn = CustomDataPredLoss3d(num_eval_points=20)
    num_epochs = 300

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        total_train_loss = train(model=model, optimizer=optimizer, dataloader=trainloader,
                loss_fn=loss_fn, scheduler=scheduler, f_values=train_f_values,
                f_meshes=train_f_meshes, domain=domain)
        total_test_loss = test(model=model, dataloader=testloader, loss_fn=loss_fn, 
            f_values=test_f_values, f_meshes=test_f_meshes, domain=domain)
            
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
        print("Warning: " + model_dir + " already exists.")

    log_file_name = model_dir + "main_info.txt"
    with open(log_file_name, "w") as f: 
        f.write('Domain: ' + ', '.join(map(str,domain)) + "\n")
        f.write('Train Batch Size: ' + str(training_bs) + "\n")
        f.write('Test Batch Size: ' + str(test_bs) + "\n")
        f.write('Model Hidden Channels: ' + str(hidden_channels) + "\n")
        f.write('Number of Training Epochs: ' + str(num_epochs) + "\n")
        f.write('Optimizer Learning Rate: ' + str(lr) + "\n")
        f.write('Optimizer Weight Decay: ' + str(weight_decay) + "\n")
        f.write('Scheduler Step Size: ' + str(step_size) + "\n")
        f.write('Scheduler Gamma: ' + str(gamma) + "\n")
        f.write('Final Total Train Loss: ' + str(total_train_loss) + "\n")
        f.write('Final Total Test Loss: ' + str(total_test_loss) + "\n")
    
    torch.save(model.state_dict(), model_dir +  "model.pth")
    print("Training complete. Saved model to " + model_dir + "model.pth.")




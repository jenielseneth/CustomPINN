import os
import torch
from typing import Type
from torch.optim.lr_scheduler import StepLR
from training_utils import GreensTrainer
from dataset_utils import GreenPINNDataset
from constants_utils import Hyperparameters
from PINN import CustomPINN_Green2D, CustomPINN_Green2D_Baseline
from PINN_2 import CustomPINN_Green2D_2
from loss import MAPELoss
import wandb


if __name__ == "__main__":
    #User input for the folder from which we retrieve data..
    user_input = input("Enter the res folder we retrieve data from: ")
    main_dir = "./res/" + user_input + "/"
    if not os.path.exists(main_dir):
        raise IsADirectoryError(f'The directory {main_dir} does not exist.')
    
    #Get data directory
    data_dir = main_dir + "data/"

        #Hyperparameters
    config = Hyperparameters(
        training_batch_size=256,
        test_batch_size=128,
        train_excl_boundary_points=True,
        test_excl_boundary_points=False,
        model_cls=CustomPINN_Green2D_Baseline,
        model_params={"hidden_size":64, "num_layers":5},
        optimizer_cls=torch.optim.Adam,
        optimizer_params={"lr":1e-2, "weight_decay":1e-4},
        scheduler_cls= None,
        scheduler_params={"step_size":0, "gamma":0.5},
        l_weights=False,
        num_epochs=30,
        num_runs=3,
    )
    # Get Datasets
    train_data = GreenPINNDataset(data_file_path=data_dir, data_file_name="data_train.pt")
    if config.train_excl_boundary_points:
        train_data.interior_points_dataset()
    test_data = GreenPINNDataset(data_file_path=data_dir, data_file_name="data_test.pt")
    if config.test_excl_boundary_points:
        test_data.interior_points_dataset()
        
    # Define loss functions
    train_loss_fn = torch.nn.MSELoss()
    test_loss_fn = [torch.nn.MSELoss(), torch.nn.L1Loss(), MAPELoss()]

    # If debug_mode is True, no model will be saved or be logged into WandB.
    user_input = input("Debug mode? No model will be saved or be logged into WandB. (y/n): ").strip().lower()
    assert user_input == "n" or user_input == "y", "You must give either the symbol 'y' or 'n'."
    debug_mode = user_input == 'y'

    ##### Refactor to pass classes and params instead of instantiations

    trainer = GreensTrainer(model_cls=config.model_cls, model_params=config.model_params,
                      optimizer_cls=config.optimizer_cls, optimizer_params=config.optimizer_params,
                      scheduler_cls=config.scheduler_cls, scheduler_params=config.scheduler_params, 
                      training_data=train_data, test_data=test_data,
                      train_loss_fn=train_loss_fn, test_loss_fn=test_loss_fn,
                      config=config, boundary_loss=True,
                      debug_mode=debug_mode)
    
    metrics = trainer.run(main_dir=main_dir)



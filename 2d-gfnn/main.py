import os
import torch
from typing import Type
from torch.optim.lr_scheduler import StepLR
from training_utils import GreensTrainer
from dataset_utils import GreenPINNDataset
from constants_utils import Hyperparameters
from PINN import CustomPINN_Green2D
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
        hidden_channels=64,
        num_layers=5,
        num_epochs=1,
        learning_rate=1e-2,
        weight_decay=1e-4,
        step_size=0,
        gamma=0.5,
        l_weights=False,
        num_runs=2
    )

    # Get Datasets
    train_data = GreenPINNDataset(data_file_path=data_dir, data_file_name="data_train.pt")
    if config.train_excl_boundary_points:
        train_data.interior_points_dataset()
    test_data = GreenPINNDataset(data_file_path=data_dir, data_file_name="data_test.pt")
    if config.test_excl_boundary_points:
        test_data.interior_points_dataset()
        
    # Initialize model
    # model = CustomPINN_Green2D(hidden_size=config.hidden_channels, num_layers=config.num_layers, domain=train_data.constants.domain, l_weights=config.l_weights)
    # model = CustomPINN_Green2D_2(hidden_size=config.hidden_channels, num_layers=config.num_layers, domain=train_data.constants.domain, l_weights=config.l_weights)
    model_cls = CustomPINN_Green2D
    model_params = {"hidden_size":config.hidden_channels, "num_layers":config.num_layers, "domain":train_data.constants.domain, "l_weights":config.l_weights}
    # Define loss functions
    train_loss_fn = torch.nn.MSELoss()
    test_loss_fn = [torch.nn.MSELoss(), torch.nn.L1Loss(), MAPELoss()]

    # Initialize optimizer and scheduler
    # optimizer = torch.optim.Adam(params=model.parameters(), )
    # scheduler = StepLR(optimizer, step_size=config.step_size, gamma=config.gamma) if config.step_size > 0 else None
    optimizer_cls = torch.optim.Adam
    optimizer_params = {"lr":config.learning_rate, "weight_decay":config.weight_decay}
    scheduler_cls = StepLR if config.step_size > 0 else None
    scheduler_params = {"step_size":config.step_size, "gamma":config.gamma}

    # If debug_mode is True, no model will be saved or be logged into WandB.
    debug_mode = input("Debug mode? No model will be saved or be logged into WandB. (y/n): ").strip().lower() == 'y'

    if not debug_mode:
        # WandB
        wandb_runner = wandb.init(
            entity="jens1225-eth-zrich",
            group="loss-averaging-test",
            project="Green2D-ConvergenceTest",
        )
    else:
        wandb_runner = None
    ##### Refactor to pass classes and params instead of instantiations

    trainer = GreensTrainer(model_cls=CustomPINN_Green2D, model_params=model_params,
                      optimizer_cls=optimizer_cls, optimizer_params=optimizer_params,
                      scheduler_cls=scheduler_cls, scheduler_params=scheduler_params, 
                      training_data=train_data, test_data=test_data,
                      train_loss_fn=train_loss_fn, test_loss_fn=test_loss_fn,
                      hyperparameters_config=config,
                      l_weights=config.l_weights,
                      boundary_loss=True,
                      debug_mode=debug_mode)
    
    metrics = trainer.run(main_dir=main_dir, config=config, wandb_run=wandb_runner)

    if not debug_mode:
        wandb_runner.finish()



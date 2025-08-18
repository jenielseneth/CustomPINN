import argparse
import os
import torch
import itertools
from typing import Type
from torch.optim.lr_scheduler import StepLR
from training_utils import GreensTrainer
from dataset_utils import GreenPINNDataset
from constants_utils import Hyperparameters
from PINN import CustomPINN_Green2D, CustomPINN_Green2D_Fourier_Dot, CustomPINN_Green2D_PoissonExplicit_W_Log, CustomPINN_Green2D_PoissonExplicit, CustomPINN_Green2D_PoissonExplicit_Dot
from loss import MAPELoss
from random_utils import retrieve_dict_from_json
from tqdm import tqdm
import logging


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO,
    format="%(filename)s:%(lineno)d - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--rd', type=str, required=True, help='Which res folder to use.')
    # Which tests to run
    parser.add_argument('--debug', action='store_true', help='Run debug mode.') 
    parser.add_argument('--wandb', type=str, help='Which name the WandB run uses.')
    parser.add_argument('--pretrain_dir', type=str, help='Directory to retrieve pretrained model from.')
    args = parser.parse_args()
    #Hyperparameters
    config = Hyperparameters(
        training_batch_size=128,
        test_batch_size=128,
        train_excl_boundary_points=False,
        test_excl_boundary_points=False,
        train_subset_idx=10000,
        test_subset_idx=4000,
        model_cls=CustomPINN_Green2D_PoissonExplicit,
        model_params={"hidden_size":16, "num_layers":5},
        optimizer_cls=torch.optim.Adam,
        optimizer_params={"lr":1e-3, "weight_decay":0},
        scheduler_cls= None,
        scheduler_params={"step_size":0, "gamma":0.5},
        l_weights=False,
        boundary_loss=True,
        harmonic_psi_loss=False,
        num_epochs=40,
        num_runs=1,
        harmonic_psi_loss_factor=0.01,
        prediction_loss_factor=1.0,
        boundary_loss_factor=1.0,
        device=torch.device("mps"),
        wandb_project_name=args.wandb if args.wandb else "test"
    )

    # User input for the folder from which we retrieve data..
    directory_input = args.rd
    if "," in directory_input:
        dirs = [directory.strip() for directory in directory_input.split(",")]
    else:
        dirs = [directory_input.strip()] 

    # If debug_mode is True, no model will be saved or be logged into WandB.
    debug_mode = args.debug

    if args.pretrain_dir is not None:
            print("Retrieving pretrained model from: ", args.pretrain_dir)
            pretrain_config_dict = retrieve_dict_from_json(args.pretrain_dir + "config.json")
            pretrain_config = Hyperparameters(**pretrain_config_dict)
    else:
        pretrain_config = None
    
    for directory in dirs:

        main_dir = "./res/" + directory + "/"
        if not os.path.exists(main_dir):
            raise IsADirectoryError(f'The directory {main_dir} does not exist.')
        
        #Get data directory
        data_dir = main_dir + "data/"

        # Get Datasets
        train_data = GreenPINNDataset(data_file_path=data_dir, data_file_name="data_train.pt", subset_idx=config.train_subset_idx)
        if config.train_excl_boundary_points:
            train_data.interior_points_dataset()
        test_data = GreenPINNDataset(data_file_path=data_dir, data_file_name="data_test.pt", subset_idx=config.test_subset_idx)
        if config.test_excl_boundary_points:
            test_data.interior_points_dataset()
            
        # Define loss functions
        train_loss_fn = torch.nn.MSELoss()
        test_loss_fn = [torch.nn.MSELoss(), torch.nn.L1Loss()]

        trainer = GreensTrainer(
                        training_data=train_data, test_data=test_data,
                        train_loss_fn=train_loss_fn, test_loss_fn=test_loss_fn,
                        config=config,
                        pretrained_model_config=pretrain_config,
                        pretrained_model_dir=args.pretrain_dir,
                        debug_mode=debug_mode)
        
        metrics = trainer.run(directory=main_dir)



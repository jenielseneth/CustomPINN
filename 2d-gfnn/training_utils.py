import torch, os, shutil
from math import prod
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
from typing import Type

import wandb
from data_generation_utils import sample_points, gcd_chebyshev_mesh_size
from plot_utils import plot_points
from pde_utils import InferenceUtils, evaluate_greens_function_integral, greens_function_laplacian_2d
from datetime import datetime
from dataset_utils import GreenPINNDataset, GreensConstantsDataclass, get_interior_boundary_idx, get_non_corners_mesh, get_corners_idx
from constants_utils import BoundaryPointLossParams, Hyperparameters
from random_utils import log_dict_as_json


class GreensTrainer:
    def __init__(self, 
                 model_cls : Type[torch.nn.Module], model_params: dict, 
                 optimizer_cls: Type[torch.optim.Optimizer], optimizer_params: dict,
                 training_data: GreenPINNDataset, test_data: GreenPINNDataset, 
                 train_loss_fn: _Loss, test_loss_fn: _Loss | list[_Loss], 
                 config: Hyperparameters,
                 scheduler_cls: Type[_LRScheduler] | None, scheduler_params: dict,
                 boundary_loss_params: BoundaryPointLossParams = None,
                 debug_mode: bool = False):
        
        '''
        Wrapper to run training and testing for Greens function models.

        Parameters:
            model_cls: Type[torch.nn.Module]
                Class type for lazy instantiating model.
            model_params: dict
                Dictionary of parameters to be passed to model_cls for instantiation.
            optimizer_cls: Type[torch.optim.Optimizer]
                Class type for lazy instantiating optimizer.
            optimizer_params: dict
                Dictionary of parameters to be passed to optimizer_cls for instantiation.
            training_data: GreenPINNDataset
                Training data using GreenPINNDataset wrapper.
            test_data: GreenPINNDataset
                Test data using GreenPINNDataset wrapper.
            train_loss_fn: _Loss
                Train loss function from torch.
            test_loss_fn: _Loss | list[_Loss]
                Test loss function / list of test loss functions from torch. Passing a list of loss functions will calculate the test loss for each of the list elements.
            config: Hyperparameters
                Hyperparameters of the run.
            scheduler_cls: Type[_LRScheduler] | None
                Class type for lazy instantiating scheduler.
            scheduler_params: dict
                Dictionary of parameters to be passed to scheduler_cls for instantiation.
            boundary_loss_params: BoundaryPointLossParams
                Optional parameters for generating boundary points for the boundary loss.
            debug_mode: bool
                If true, the Trainer doesn't log results to WandB or saves a model.




            
        '''
        
        ## Lazy instantiation of model, optimizer and scheduler for the purpose of multiple runs.
        self.model_cls = model_cls
        self.model_params = model_params
        self.optimizer_cls = optimizer_cls
        self.optimizer_params = optimizer_params
        self.scheduler_cls = scheduler_cls
        self.scheduler_params = scheduler_params

        self.train_loss_fn = train_loss_fn
        self.test_loss_fn = test_loss_fn if isinstance(test_loss_fn, list) else [test_loss_fn]
        self.training_data = training_data
        self.test_data = test_data
        self.config = config
        self.trainloader = DataLoader(self.training_data, batch_size=self.config.training_batch_size, shuffle=True)
        self.testloader = DataLoader(self.test_data, batch_size=self.config.test_batch_size, shuffle=True)
        self.inference_utils = InferenceUtils(constants=self.training_data.constants, config=config)
        self.boundary_loss = config.boundary_loss

            
        # If true, the model calculates the additional loss term ||G(x,y) - boundary conditions||.

        # Temporary solution to calculate boundary points at each epoch to guarantee boundary loss is calculated.
        if self.boundary_loss:
            self.bnd_points_size = (20, 20) if boundary_loss_params is None else boundary_loss_params["bnd_points_size"]
            self.domain_mesh_size = gcd_chebyshev_mesh_size(self.bnd_points_size) if boundary_loss_params is None else boundary_loss_params["domain_mesh_size"]
            self.bnd_points = sample_points(domain=self.training_data.constants.domain, mesh_size=self.bnd_points_size, mesh_type="chebyshev", boundary=True)
            self.bnd_points = get_non_corners_mesh(domain=self.training_data.constants.domain, mesh=self.bnd_points)[:, None, :].expand(-1, self.domain_mesh_size[0]*self.domain_mesh_size[1], -1).to(config.device)
            self.domain_mesh = sample_points(domain=self.training_data.constants.domain, mesh_size=self.domain_mesh_size, mesh_type="chebyshev")[None, :, :].expand(len(self.bnd_points), -1, -1).to(config.device)
        
        self.debug_mode = debug_mode
        self.device = self.config.device

        # Establish device
        self.inference_utils.to_device(self.device)
        self.training_data.constants.to_device(self.device)
        self.test_data.constants.to_device(self.device)


    def _train(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> torch.Tensor: 
        size = len(self.trainloader.dataset)
        model.train()
        current_num = 0
        total_loss = 0
        
        bar = tqdm(enumerate(self.trainloader), desc="Training", total=len(self.trainloader), leave=False, ascii=' >=')
        for _, item in bar:

            #Temporary solution to avoid calculating on corners

            # Check if training data has excluded boundary points
            if self.config.train_excl_boundary_points == False:
                n_c, _ = get_corners_idx(domain=self.training_data.constants.domain, mesh=item["crd"])
                evaluation_mesh = item["crd"][n_c].to(self.device)
                integration_mesh_values = item["f_vals"][n_c].to(self.device)
                u_gt = item["u_vals"][n_c].to(self.device)
            else:
                u_gt = item["u_vals"].to(self.device)
                integration_mesh_values = item["f_vals"].to(self.device)
                evaluation_mesh = item["crd"].to(self.device)

            ### COMPUTE PREDICTION AND LOSS
            # Compute ||∫G(x,y)f(y)dy - u(x)||
            u_prediction = evaluate_greens_function_integral(greens_function=model, integration_mesh_values=integration_mesh_values, 
                                                             evaluation_mesh=evaluation_mesh, integration_mesh=self.training_data.constants.integration_mesh, 
                                                             quadrature_weights=self.inference_utils.quadrature_weights)

            loss = self.train_loss_fn(u_prediction, u_gt)
            
            # Temporary solution to calculate the Laplacian of Psi(x, y) for Poisson equation.
            int_mesh_shape = self.training_data.constants.integration_mesh.shape
            eval_mesh_shape = evaluation_mesh.shape
            # x_laplacian = evaluation_mesh[:, None, :].expand(-1, int_mesh_shape[0], -1)  # b x f x 2 Tensor 
            # x_laplacian.requires_grad = True
            # y_laplacian = self.training_data.constants.integration_mesh[None, :, :].expand(eval_mesh_shape[0], -1, -1) # b x f x 2 Tensor 
            laplacian_loss = greens_function_laplacian_2d(
                greens_function= lambda x, y: model.psi(x[:, None, :].expand(-1, int_mesh_shape[0], -1), y[None, :, :].expand(eval_mesh_shape[0], -1, -1))[0],
                delta_function_center=evaluation_mesh, 
                y=self.training_data.constants.integration_mesh
                )
            print(laplacian_loss)
            assert False
            loss += laplacian_loss

            # Calculate boundary loss
            if self.boundary_loss == True:
                assert self.bnd_points.shape == self.domain_mesh.shape, f"Boundary points ({self.bnd_points.shape}) and domain mesh ({self.domain_mesh.shape}) must have the same batch size."

                # Calculate ||G(x,y) - boundary conditions||
                greens_function_boundary_eval = model(self.bnd_points, self.domain_mesh)
                boundary_loss_term = self.train_loss_fn(greens_function_boundary_eval, torch.zeros_like(greens_function_boundary_eval))
                loss += boundary_loss_term

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = len(item["crd"])
            current_num += batch_size
            tqdm.write(f"\rAvg Train Loss per sample: {loss / batch_size :>7f}  [{current_num:>5d}/{size:>5d}] \n", end="")
            total_loss += loss

        return total_loss
    
    def _test(self, model) -> torch.Tensor:
        size = len(self.testloader.dataset)
        model.eval()
        test_loss = torch.zeros(len(self.test_loss_fn)).to(self.device)
        with torch.no_grad():
            for item in self.testloader:
                #Temporary solution to avoid calculating on corners 

                # Check if test data has excluded boundary points
                if not self.config.test_excl_boundary_points:
                    n_c, _ = get_corners_idx(domain=self.test_data.constants.domain, mesh=item["crd"])
                    eval_mesh = item["crd"][n_c].to(self.device)
                    integration_mesh_values = item["f_vals"][n_c].to(self.device)
                    u_gt = item["u_vals"][n_c].to(self.device)
                else:
                    eval_mesh = item["crd"].to(self.device)
                    integration_mesh_values = item["f_vals"].to(self.device)
                    u_gt = item["u_vals"].to(self.device)

                u_prediction = evaluate_greens_function_integral(greens_function=model, integration_mesh_values=integration_mesh_values, 
                                                                 evaluation_mesh=eval_mesh, integration_mesh=self.test_data.constants.integration_mesh,
                                                                 quadrature_weights=self.inference_utils.quadrature_weights)
                loss = torch.tensor([loss_fn(u_prediction, u_gt) for loss_fn in self.test_loss_fn]).to(self.device)
                test_loss += loss

        for i, tl in enumerate(test_loss):
            tqdm.write(f"Avg Test Loss {self.test_loss_fn[i]} per sample: {tl / size :>8f} \n", end="")
        return test_loss
    


    def run(self, directory: list[str] | str):
        '''
        Runs the training and testing loops.
        :param main_dir: Main directory to source data from and store models in.
        '''

        try:
            if self.debug_mode:
                print("Debug mode is enabled. No model will be saved.")
            else:
                #Get WandB Project Name
                # wandb_project_name_input = input("Name of WandB project? Press enter for project='test': ")
                # wandb_project_name_input = "100fPINNposexpl_fem"
                wandb_project_name_input = ""
                if not wandb_project_name_input == "":
                    project_name = wandb_project_name_input
                    model_dir = directory + f"models/{project_name}/" 

                else:
                    project_name = "test"
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_dir = directory + f"models/model_{timestamp}/" 
                
                # Check model_dir doesn't exist to prevent overwriting.
                try:
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                except OSError:
                    print("Warning: " + model_dir + " already exists.")
                    raise
            
            # Store best test loss and the associated train loss.
            best_test_loss = torch.tensor([float('inf') for _ in range(len(self.test_loss_fn))]).to(self.config.device)
            associated_train_loss = torch.zeros_like(best_test_loss).to(self.config.device)
            best_train_loss = torch.tensor([float('inf')]).to(self.config.device)


            # RUNS
            for i_run in tqdm(range(self.config.num_runs), desc=f"Training Runs", ascii="░▒█", leave=True):
                
                if not self.debug_mode:
                    # WandB initialization
                    wandb_config = {
                            **self.config.__dict__,
                            **{k: v for k, v in self.training_data.constants.__dict__.items() if k != 'integration_mesh'},
                        }

                    wandb_run = wandb.init(
                        entity="jens1225-eth-zrich",
                        project=project_name,
                        config=wandb_config
                    )
                else: 
                    wandb_run = None
                    
                # Model, optimizer, scheduler initialization. 
                model = self.model_cls(**self.model_params)
                model.to(self.device)

                optimizer = self.optimizer_cls(params=model.parameters(), **self.optimizer_params)
                if self.scheduler_cls is not None:
                    scheduler = self.scheduler_cls(optimizer = optimizer, **self.scheduler_params)
                else:
                    scheduler = None

                if self.config.num_runs == 1:
                    tqdm.write("Running trainer.")
                else:
                    tqdm.write(f"Running trainer in run: {i_run}.")

                # Loop through the epochs and train the model.
                for epoch in tqdm(range(self.config.num_epochs), desc="Epochs", leave=False):
                    train_loss = self._train(model=model, optimizer=optimizer)
                    test_loss = self._test(model)

                    if scheduler is not None:
                        scheduler.step()
                    
                    # Log best train / test losses
                    if 0 <= epoch:
                        # Best train loss log
                        best_train_loss = train_loss if train_loss < best_train_loss else best_train_loss

                        # Best test loss log
                        best_indices = torch.nonzero(test_loss < best_test_loss).squeeze()
                        if best_indices.dim() > 0:
                            for i in best_indices:
                                best_test_loss[i] = test_loss[i]
                                associated_train_loss[i] = train_loss
                                if not self.debug_mode:
                                    tqdm.write(f"New best model found at epoch {epoch+1} with test loss {best_test_loss[i]}. Saving model.")
                                    torch.save(model.state_dict(), model_dir + f"model_best_{self.test_loss_fn[i]}.pth")


                    ## Log metrics
                    total_test_metrics = {f"test/total_{self.test_loss_fn[i]}": test_loss[i].item() for i in range(len(self.test_loss_fn))}
                    total_metrics = {f"train/total_{self.train_loss_fn}": train_loss.item(), **total_test_metrics}
                    test_int_mesh_size = self.test_data.constants.integration_mesh_size
                    train_int_mesh_size = self.training_data.constants.integration_mesh_size
                    res_inv_test_metrics = {f"test/int_mesh_resolution_norm_{self.test_loss_fn[i]}": test_loss[i].item()/(prod(test_int_mesh_size))for i in range(len(self.test_loss_fn))}
                    res_inv_metrics = {f"train/int_mesh_resolution_norm_{self.train_loss_fn}": train_loss.item()/prod(train_int_mesh_size), **res_inv_test_metrics}
                    metrics = {**total_metrics, **res_inv_metrics}
                    
                    # Log best losses at last epoch
                    if epoch == self.config.num_epochs - 1:
                        if not self.debug_mode:
                            # Storing test loss 
                            for i, loss_fn in enumerate(self.test_loss_fn):
                                metrics[f'best/{loss_fn}'] = best_test_loss[i].item()
                                metrics[f'best/{loss_fn}_assoc_train_loss'] = associated_train_loss[i].item()
                                metrics[f'best/train_loss'] = best_train_loss.item()
                    
                    # Log metrics
                    if wandb_run is not None:
                        wandb_run.log({**metrics})
                
                # Finish WandB log per run.
                if wandb_run is not None:
                    wandb_run.finish()
            
            # End of training loop
            
            # Save config.json and final model.
            if not self.debug_mode:
                log_dict_as_json(self.config.get_dict(), model_dir + 'config.json')
                torch.save(model.state_dict(), model_dir +  "model_final.pth")
                tqdm.write("Training complete. Saved final model to " + model_dir + "model_final.pth.")
            return {"best_test_loss": best_test_loss, "best_train_loss": associated_train_loss, "final_test_loss": test_loss, "final_train_loss": train_loss}


        except Exception:
            if not self.debug_mode:
                # Delete saved intermediate model directory. 
                shutil.rmtree(model_dir)
                print(f"Error occurred during training. Removed directory {model_dir}.")
            raise
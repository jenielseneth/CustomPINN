import torch, os, shutil
from math import prod
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
from typing import Type
from chebyshev_utils import cheb_2d_impl

import wandb
from data_generation_utils import sample_points, gcd_chebyshev_mesh_size
from plot_utils import plot_points
from pde_utils import InferenceUtils, evaluate_greens_function_integral, greens_function_laplacian_2d, u_laplacian_2d
from datetime import datetime
from dataset_utils import GreenPINNDataset, GreensConstantsDataclass, get_interior_boundary_idx, get_interior_mesh, get_non_corners_mesh, get_corners_idx
from constants_utils import BoundaryPointLossParams, Hyperparameters
from random_utils import log_dict_as_json


class GreensTrainer:
    def __init__(self, 
                 training_data: GreenPINNDataset, test_data: GreenPINNDataset, 
                 train_loss_fn: _Loss, test_loss_fn: _Loss | list[_Loss], 
                 config: Hyperparameters,
                 pretrained_model_dir: str = None,
                 pretrained_model_config: Hyperparameters = None,
                 boundary_loss_fn: _Loss = None,
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
        self.pretrained_model_config = pretrained_model_config
        self.pretrained_model_dir = pretrained_model_dir
        self.config = config

        if self.pretrained_model_config is not None:
            assert self.pretrained_model_dir is not None, "If pretrained_model_config is given, pretrained_model_dir must also be given."
            self.model_cls = self.pretrained_model_config.model_cls
            self.model_params = self.pretrained_model_config.model_params
            self.optimizer_cls = self.pretrained_model_config.optimizer_cls
            self.optimizer_params = self.pretrained_model_config.optimizer_params
            self.scheduler_cls = self.pretrained_model_config.scheduler_cls
            self.scheduler_params = self.pretrained_model_config.scheduler_params
        else:
            self.model_cls = self.config.model_cls
            self.model_params = self.config.model_params
            self.optimizer_cls = self.config.optimizer_cls
            self.optimizer_params = self.config.optimizer_params
            self.scheduler_cls = self.config.scheduler_cls
            self.scheduler_params = self.config.scheduler_params

        self.train_loss_fn = train_loss_fn
        self.boundary_loss_fn = boundary_loss_fn if boundary_loss_fn is not None else self.train_loss_fn
        self.test_loss_fn = test_loss_fn if isinstance(test_loss_fn, list) else [test_loss_fn]
        self.training_data = training_data
        self.test_data = test_data
        self.trainloader = DataLoader(self.training_data, batch_size=self.config.training_batch_size, shuffle=True)
        self.testloader = DataLoader(self.test_data, batch_size=self.config.test_batch_size, shuffle=True)
        self.inference_utils = InferenceUtils(constants=self.training_data.constants, config=config)
        self.boundary_loss = config.boundary_loss

        # Temporary solution to calculate boundary points at each epoch to guarantee boundary loss is calculated.
        if self.boundary_loss:
            self.bnd_points = sample_points(domain=self.training_data.constants.domain, 
                                            mesh_size=(128, 128), 
                                            mesh_type=self.training_data.constants.evaluation_mesh_type, 
                                            boundary=True)
            self.bnd_points = get_non_corners_mesh(domain=self.training_data.constants.domain, mesh=self.bnd_points).to(config.device) # b x 2 Tensor
        
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
        total_prediction_loss = 0
        total_harmonic_psi_loss = 0
        # total_harmonic_u_loss = 0
        total_boundary_loss = 0
        
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
            prediction_loss = self.config.prediction_loss_factor * self.train_loss_fn(u_prediction, u_gt)
            loss = prediction_loss
            

            # Temporary solution to calculate the Laplacian of Psi(x, y) for Poisson equation.
            if self.config.harmonic_psi_loss:
                
                # Define the psi function to be used for the harmonic loss.
                def psi(x, s):
                    # Expand x and s to match the expected input shape
                    if x.dim() == 2:
                        assert s.dim() == 2, "s must be a 2D tensor if x is 2D."
                        x = x[:, None, :].expand(-1, s.shape[0], -1)  # b x f x 2 Tensor
                        s = s[None, :, :].expand(x.shape[0], -1, -1)  # b x f x 2 Tensor

                    return model.psi(x, s)[...,0]
                
                harmonic_psi_term = greens_function_laplacian_2d(
                    greens_function=psi,
                    x=evaluation_mesh[0:4], 
                    s=self.training_data.constants.integration_mesh
                    )
                harmonic_psi_loss = (harmonic_psi_term**2).mean()
                loss += self.config.harmonic_psi_loss_factor * harmonic_psi_loss
            else:
                harmonic_psi_loss = torch.tensor(0.0, device=self.device)

            
            # # Temporary solution to calculate the Laplacian of u(x, y) for Poisson equation.
            # if self.config.harmonic_u_loss:
            #     sample_evals = evaluation_mesh[0:4]
            #     harmonic_u_loss = torch.tensor(0.)

            #     for i in range(4):

            #         assert self.training_data.constants.integration_mesh_type == "chebyshev", "Current implementation relies on integration mesh being chebyshev."
                
            #         eval_mesh = get_interior_mesh(domain=self.training_data.constants.domain, mesh=sample_evals[i])

            #         f_eval_point_values = cheb_2d_impl(eval_points=eval_mesh,
            #                     chebyshev_size=self.training_data.constants.integration_mesh_size,
            #                     chebyshev_values=integration_mesh_values[i],
            #                     domain=self.training_data.constants.domain
            #                     )
                    
            #         grad_2_u = u_laplacian_2d(
            #             greens_function=model,
            #             x=eval_mesh, 
            #             s=self.training_data.constants.integration_mesh,
            #             s_values=integration_mesh_values[i],
            #             quadrature_weights=self.inference_utils.quadrature_weights
            #             )
                    
            #         harmonic_u_loss += torch.nn.functional.mse_loss(grad_2_u,f_eval_point_values)
            #         loss += self.config.harmonic_u_loss_factor * harmonic_u_loss
            # else:
            #     harmonic_u_loss = torch.tensor(0.0, device=self.device)

            # Calculate boundary loss ||G(x,y) - boundary conditions||
            if self.boundary_loss == True:
                greens_function_boundary_eval = model(self.bnd_points[:, None, :].expand(-1, self.training_data.constants.integration_mesh.shape[0] ,-1), 
                                                      self.training_data.constants.integration_mesh[None, ...].expand(self.bnd_points.shape[0], -1, -1))
                boundary_loss = self.boundary_loss_fn(greens_function_boundary_eval, torch.zeros_like(greens_function_boundary_eval))
                loss += self.config.boundary_loss_factor * boundary_loss
            else: 
                boundary_loss = torch.tensor(0.0, device=self.device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = len(item["crd"])
            current_num += batch_size
            tqdm.write(f"\rAvg Train Loss Per Sample: {loss / batch_size :>9f} | Predict Loss: {prediction_loss :>9f} | Harmonic Psi Loss: {harmonic_psi_loss :>9f} | Boundary Loss: {boundary_loss :>9f} |   [{current_num:>5d}/{size:>5d}] \n", end="")
            total_loss += loss
            total_prediction_loss += prediction_loss
            total_harmonic_psi_loss += harmonic_psi_loss
            total_boundary_loss += boundary_loss
            # total_harmonic_u_loss += harmonic_u_loss

        return total_loss, total_prediction_loss, total_boundary_loss, total_harmonic_psi_loss
    
    def _test(self, model) -> torch.Tensor:
        size = len(self.testloader.dataset)
        model.eval()
        test_loss = torch.zeros(len(self.test_loss_fn)).to(self.device)
        total_prediction_loss = 0
        total_harmonic_psi_loss = 0
        total_boundary_loss = 0
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
            
            # Define the psi function to be used for the harmonic Psi loss.
            def psi(x, s):
                # Expand x and s to match the expected input shape
                if x.dim() == 2:
                    assert s.dim() == 2, "s must be a 2D tensor if x is 2D."
                    x = x[:, None, :].expand(-1, s.shape[0], -1)  # b x f x 2 Tensor
                    s = s[None, :, :].expand(x.shape[0], -1, -1)  # b x f x 2 Tensor

                return model.psi(x, s)[...,0]
            
            # Harmonic loss 
            harmonic_psi_term = greens_function_laplacian_2d(
                greens_function=psi,
                x=eval_mesh[0:4], 
                s=self.training_data.constants.integration_mesh
                )
            harmonic_psi_loss = torch.tensor([loss_fn(harmonic_psi_term, torch.zeros_like(harmonic_psi_term)) for loss_fn in self.test_loss_fn], device=self.device)
            test_loss += self.config.harmonic_psi_loss_factor * harmonic_psi_loss

            with torch.no_grad():
                # Prediction loss
                u_prediction = evaluate_greens_function_integral(greens_function=model, integration_mesh_values=integration_mesh_values, 
                                                                    evaluation_mesh=eval_mesh, integration_mesh=self.test_data.constants.integration_mesh,
                                                                    quadrature_weights=self.inference_utils.quadrature_weights)
                prediction_loss = torch.tensor([loss_fn(u_prediction, u_gt) for loss_fn in self.test_loss_fn], device=self.device)
                test_loss += self.config.prediction_loss_factor * prediction_loss

                # Boundary loss ||G(x,y) - boundary conditions||
                greens_function_boundary_eval = model(self.bnd_points[:, None, :].expand(-1, self.training_data.constants.integration_mesh.shape[0] ,-1), 
                                                    self.training_data.constants.integration_mesh[None, ...].expand(self.bnd_points.shape[0], -1, -1))
                boundary_loss = torch.tensor([loss_fn(greens_function_boundary_eval, torch.zeros_like(greens_function_boundary_eval)) for loss_fn in self.test_loss_fn], device=self.device)
                test_loss += self.config.boundary_loss_factor * boundary_loss
                total_prediction_loss += prediction_loss
                total_harmonic_psi_loss += harmonic_psi_loss
                total_boundary_loss += boundary_loss

        for i, tl in enumerate(test_loss):
            tqdm.write(f"Avg Test Loss {self.test_loss_fn[i]} per sample: {tl / size :>8f} \n", end="")
        return test_loss, total_prediction_loss, total_harmonic_psi_loss, total_boundary_loss
    


    def run(self, directory: list[str] | str):
        '''
        Runs the training and testing loops.
        :param main_dir: Main directory to source data from and store models in.
        '''

        try:
            # Store best test loss and the associated train loss.
            best_test_loss = torch.tensor([float('inf') for _ in range(len(self.test_loss_fn))], device=self.config.device)
            best_test_prediction_loss = torch.tensor([float('inf') for _ in range(len(self.test_loss_fn))], device=self.config.device)
            best_train_loss = torch.tensor([float('inf')], device=self.config.device)

            # RUNS
            for i_run in tqdm(range(self.config.num_runs), desc=f"Training Runs", ascii="░▒█", leave=True):
                
                if not self.debug_mode:

                    project_name = self.config.wandb_project_name
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

                    
                    # Generate model directory
                    if i_run == 0:
                        if self.debug_mode:
                            print("Debug mode is enabled. No model will be saved.")
                        else:
                            #Get WandB Project Name
                            wand_b_run_name = wandb_run.name
                            model_dir = directory + f"models/{project_name}_{wand_b_run_name}/" 
                            
                            # Check model_dir doesn't exist to prevent overwriting.
                            try:
                                if not os.path.exists(model_dir):
                                    os.makedirs(model_dir)
                            except OSError:
                                print("Warning: " + model_dir + " already exists.")
                                raise


                else: 
                    wandb_run = None
                    
                # Model, optimizer, scheduler initialization. 
                if self.pretrained_model_config is not None:
                    model = self.model_cls(**self.model_params)
                    model.load_state_dict(torch.load(self.pretrained_model_dir + "model_final.pth", map_location=self.config.device))
                else:
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
                    train_loss, train_prediction_loss, train_boundary_loss, train_harmonic_psi_loss = self._train(model=model, optimizer=optimizer)
                    test_loss, test_prediction_loss, test_harmonic_psi_loss, test_boundary_loss  = self._test(model)

                    if scheduler is not None:
                        scheduler.step()
                    
                    # Log best train / test losses
                    if 0 <= epoch:
                        # Log best total training loss
                        best_train_loss = train_loss if train_loss < best_train_loss else best_train_loss
                        
                        # Log best total test loss
                        best_indices = torch.nonzero(test_loss < best_test_loss).squeeze()
                        if best_indices.dim() > 0:
                            for i in best_indices:
                                best_test_loss[i] = test_loss[i]
                                if not self.debug_mode:
                                    tqdm.write(f"New best model found at epoch {epoch+1} with test loss {best_test_loss[i]}. Saving model.")
                                    torch.save(model.state_dict(), model_dir + f"model_best_{self.test_loss_fn[i]}.pth")
                        
                        # Log best prediction test loss
                        best_indices = torch.nonzero(test_prediction_loss < best_test_prediction_loss).squeeze()
                        if best_indices.dim() > 0:
                            for i in best_indices:
                                best_test_prediction_loss[i] = test_prediction_loss[i]
                                if not self.debug_mode:
                                    tqdm.write(f"New best model found at epoch {epoch+1} with prediction test loss {best_test_prediction_loss[i]}. Saving model.")
                                    torch.save(model.state_dict(), model_dir + f"model_best_prediction_{self.test_loss_fn[i]}.pth")


                    ## Log metrics
                    total_test_metrics = {f"test/total_{self.test_loss_fn[i]}": test_loss[i].item() for i in range(len(self.test_loss_fn))}
                    total_test_metrics.update({f"test/prediction_{self.test_loss_fn[i]}": test_prediction_loss[i].item() for i in range(len(self.test_loss_fn))})
                    total_test_metrics.update({f"test/harmonic_psi_{self.test_loss_fn[i]}": test_harmonic_psi_loss[i].item() for i in range(len(self.test_loss_fn))})
                    total_test_metrics.update({f"test/boundary_{self.test_loss_fn[i]}": test_boundary_loss[i].item() for i in range(len(self.test_loss_fn))})
                    
                    total_metrics = {f"train/total_{self.train_loss_fn}": train_loss.item(), 
                                     f"train/prediction_{self.train_loss_fn}": train_prediction_loss.item(),
                                     f"train/harmonic_psi_{self.train_loss_fn}": train_harmonic_psi_loss.item(),
                                     f"train/boundary_{self.train_loss_fn}": train_boundary_loss.item(),
                                       **total_test_metrics}
                    # Calculate resolution invariant loss
                    # test_int_mesh_size = self.test_data.constants.integration_mesh_size
                    # train_int_mesh_size = self.training_data.constants.integration_mesh_size
                    # res_inv_test_metrics = {f"test/int_mesh_resolution_norm_{self.test_loss_fn[i]}": test_loss[i].item()/(prod(test_int_mesh_size))for i in range(len(self.test_loss_fn))}
                    # res_inv_metrics = {f"train/int_mesh_resolution_norm_{self.train_loss_fn}": train_loss[0].item()/prod(train_int_mesh_size), **res_inv_test_metrics}
                    metrics = {**total_metrics}
                    
                    # Log best losses at last epoch
                    if epoch == self.config.num_epochs - 1:
                        if not self.debug_mode:
                            # Storing test loss 
                            for i, loss_fn in enumerate(self.test_loss_fn):
                                metrics[f'best/{loss_fn}'] = best_test_loss[i].item()
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
            return {"best_test_loss": best_test_loss, "final_test_loss": test_loss, "final_train_loss": train_loss}


        except Exception:
            if not self.debug_mode:
                # Delete saved intermediate model directory. 
                shutil.rmtree(model_dir)
                print(f"Error occurred during training. Removed directory {model_dir}.")
            raise
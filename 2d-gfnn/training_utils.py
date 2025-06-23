import torch, os, shutil
from math import prod
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from tqdm import tqdm
from typing import Type
from data_generation_utils import sample_points, gcd_chebyshev_mesh_size
from plot_utils import plot_points
from pde_utils import evaluate_greens_function_integral
from datetime import datetime
from dataset_utils import GreenPINNDataset, get_non_corners_mesh, get_corners_idx
from constants_utils import BoundaryPointLossParams, Hyperparameters
from random_utils import log_dict_as_json

class GreensTrainer:
    def __init__(self, 
                 model_cls : Type[torch.nn.Module], model_params: dict, 
                 optimizer_cls: Type[torch.optim.Optimizer], optimizer_params: dict,
                 training_data: GreenPINNDataset, test_data: GreenPINNDataset, 
                 train_loss_fn: _Loss, test_loss_fn: _Loss | list[_Loss], 
                 hyperparameters_config: Hyperparameters,
                 scheduler_cls, scheduler_params: dict,
                 l_weights: bool = False, 
                 boundary_loss: bool = True, boundary_loss_params: BoundaryPointLossParams = None,
                 debug_mode: bool = False):
        
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
        self.traind_constants = self.training_data.constants
        self.test_data = test_data
        self.testd_constants = self.test_data.constants
        self.trainloader = DataLoader(self.training_data, batch_size=hyperparameters_config.training_batch_size, shuffle=True)
        self.testloader = DataLoader(self.test_data, batch_size=hyperparameters_config.test_batch_size, shuffle=True)
        self.l_weights = l_weights
        if self.l_weights:
            self.quadrature_weights = None
        else:
            self.quadrature_weights = self.training_data.constants.quadrature_weights
            assert self.quadrature_weights is not None, "If weights are not learned, the quadrature weights must be initialized."

        self.boundary_loss = boundary_loss
        if self.boundary_loss:
            self.bnd_points_size = (20, 20) if boundary_loss_params is None else boundary_loss_params["bnd_points_size"]
            self.domain_mesh_size = gcd_chebyshev_mesh_size(self.bnd_points_size) if boundary_loss_params is None else boundary_loss_params["domain_mesh_size"]
            self.bnd_points = sample_points(domain=self.traind_constants.domain, mesh_size=self.bnd_points_size, mesh_type="chebyshev", boundary=True)
            self.bnd_points = get_non_corners_mesh(domain=self.traind_constants.domain, mesh=self.bnd_points)[:, None, :].expand(-1, self.domain_mesh_size[0]*self.domain_mesh_size[1], -1)
            self.domain_mesh = sample_points(domain=self.traind_constants.domain, mesh_size=self.domain_mesh_size, mesh_type="chebyshev")[None, :, :].expand(len(self.bnd_points), -1, -1)
        
        self.debug_mode = debug_mode

    def _train(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer): 
        size = len(self.trainloader.dataset)
        model.train()
        current_num = 0
        total_loss = 0
        
        bar = tqdm(enumerate(self.trainloader), desc="Training", total=len(self.trainloader), leave=False, ascii=' >=')
        for _, item in bar:
            # Compute prediction and loss
            u_gt = item["u_vals"]
            u_prediction = evaluate_greens_function_integral(greens_function=model, integration_mesh_values=item["f_vals"], evaluation_mesh=item["crd"], dataset_constants=self.training_data.constants)
            
            loss = self.train_loss_fn(u_prediction, u_gt) 

            if self.boundary_loss:
                assert self.bnd_points.shape == self.domain_mesh.shape, f"Boundary points ({self.bnd_points.shape}) and domain mesh ({self.domain_mesh.shape}) must have the same batch size."
                
                # Calculate ||G(x,y) - boundary conditions||
                greens_function_boundary_eval = model(self.bnd_points, self.domain_mesh)
                boundary_loss_term = self.train_loss_fn(greens_function_boundary_eval, torch.zeros_like(greens_function_boundary_eval))
                loss += boundary_loss_term

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()

            batch_size = len(item["crd"])
            current_num += batch_size
            tqdm.write(f"\rAvg Train Loss per sample: {loss / batch_size :>7f}  [{current_num:>5d}/{size:>5d}] \n", end="")
            total_loss += loss

        return total_loss
    
    def _test(self, model):
        size = len(self.testloader.dataset)
        model.eval()
        test_loss = torch.zeros(len(self.test_loss_fn))
        with torch.no_grad():
            for item in self.testloader:
                #Temporary solution to avoid calculating on corners 
                n_c, _ = get_corners_idx(domain=self.testd_constants.domain, mesh=item["crd"])
                eval_mesh = item["crd"][n_c]
                integration_mesh_values = item["f_vals"][n_c]
                u_prediction = evaluate_greens_function_integral(greens_function=model, integration_mesh_values=integration_mesh_values, evaluation_mesh=eval_mesh, dataset_constants=self.test_data.constants)
                u_gt = item["u_vals"][n_c]
                loss = torch.tensor([loss_fn(u_prediction, u_gt) for loss_fn in self.test_loss_fn])
                test_loss += loss
        test_loss = torch.tensor(test_loss.tolist())

        for i, tl in enumerate(test_loss):
            tqdm.write(f"Avg Test Loss {self.test_loss_fn[i]} per sample: {tl / size :>8f} \n", end="")
        return test_loss
    
    def run(self, main_dir: str, config: Hyperparameters, wandb_run=None):
        '''
        Runs the training and testing loop for the specified number of epochs.
        :param num_epochs: Number of epochs to train the model.
        :param wandb_run: Optional wandb run object to log metrics.
        '''

        try:
            if self.debug_mode:
                print("Debug mode is enabled. No model will be saved.")
            else:
                ##Naming convention for generated model directory
                if wandb_run is not None:
                    model_dir = main_dir + "models/" + wandb_run.name + "/"
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_dir = main_dir + f"models/model_{timestamp}/" 
                try:
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                except OSError:
                    print("Warning: " + model_dir + " already exists.")
                    raise
            
            # Store best test loss and the associated train loss.
            best_test_loss = torch.tensor([float('inf') for _ in range(len(self.test_loss_fn))])
            associated_train_loss = torch.zeros_like(best_test_loss)

            # Store train and test loss trajectories for averaging purposes.
            train_loss_log = torch.zeros((config.num_runs, config.num_epochs,1))
            test_loss_log = torch.zeros((config.num_runs, config.num_epochs , len(self.test_loss_fn)))

            for i_run in tqdm(range(config.num_runs), desc=f"Training Runs", ascii="░▒█", leave=True):
                
                model = self.model_cls(**self.model_params)
                optimizer = self.optimizer_cls(params=model.parameters(), **self.optimizer_params)
                if self.scheduler_cls is not None:
                    scheduler = self.scheduler_cls(optimizer = optimizer, **self.scheduler_params)
                else:
                    scheduler = None

                if config.num_runs == 1:
                    tqdm.write("Running trainer.")
                else:
                    tqdm.write(f"Running trainer in run: {i_run}.")
                ##Loop through the epochs and train the model.
                for epoch in tqdm(range(config.num_epochs), desc="Epochs", leave=False):
                    train_loss = self._train(model=model, optimizer=optimizer)
                    test_loss = self._test(model)

                    train_loss_log[i_run, epoch] = train_loss
                    test_loss_log[i_run, epoch] = test_loss

                    if scheduler is not None:
                        scheduler.step()
                    
                    if 0 <= epoch:
                        best_indices = torch.nonzero(test_loss < best_test_loss).squeeze()
                        if best_indices.dim() > 0:
                            for i in best_indices:
                                best_test_loss[i] = test_loss[i]
                                associated_train_loss[i] = train_loss
                                if not self.debug_mode:
                                    tqdm.write(f"New best model found at epoch {epoch+1} with test loss {best_test_loss[i]}. Saving model.")
                                    torch.save(model.state_dict(), model_dir + f"model_best_{self.test_loss_fn[i]}.pth")

                    ## Log metrics
                    if i_run == config.num_runs-1: 
                        avg_test_loss = test_loss_log[:, epoch].mean(dim=0)
                        avg_train_loss = train_loss_log[:, epoch].mean(dim=0)
                        total_test_metrics = {f"test/total_{self.test_loss_fn[i]}": avg_test_loss[i] for i in range(len(self.test_loss_fn))}
                        total_metrics = {f"train/total_{self.train_loss_fn}": avg_train_loss, **total_test_metrics}
                        test_int_mesh_size = self.testd_constants.integration_mesh_size
                        train_int_mesh_size = self.traind_constants.integration_mesh_size
                        res_inv_test_metrics = {f"test/int_mesh_resolution_norm_{self.test_loss_fn[i]}": avg_test_loss[i]/(prod(test_int_mesh_size))for i in range(len(self.test_loss_fn))}
                        res_inv_metrics = {f"train/int_mesh_resolution_norm_{self.train_loss_fn}": avg_train_loss/prod(train_int_mesh_size), **res_inv_test_metrics}
                        metrics = {**total_metrics, **res_inv_metrics}
                        
                        # Log best loss
                        if epoch == config.num_epochs - 1:
                            if not self.debug_mode:
                                # Storing test loss 
                                for i, loss_fn in enumerate(self.test_loss_fn):
                                    metrics[f'best/{loss_fn}'] = best_test_loss[i]
                                    metrics[f'best/{loss_fn}_assoc_train_loss'] = associated_train_loss[i]
                        if wandb_run is not None:
                            wandb_run.log({**metrics})
                ## End of training loop
            
            if not self.debug_mode:
                log_dict_as_json(config.__dict__, model_dir + 'config.json')
                torch.save(model.state_dict(), model_dir +  "model_final.pth")
                tqdm.write("Training complete. Saved final model to " + model_dir + "model_final.pth.")
            return {"best_test_loss": best_test_loss, "best_train_loss": associated_train_loss, "final_test_loss": test_loss, "final_train_loss": train_loss}


        except Exception:
            if not self.debug_mode:
                shutil.rmtree(model_dir)
                print(f"Error occurred during training. Removed directory {model_dir}.")
            raise
import importlib
from typing import Literal, Type, TypedDict
from dataclasses import dataclass

import sympy as sp
import torch


mesh_type = Literal["uniform", "chebyshev", "random"]

multi_mesh_training_method = Literal["obo", "standardize"]

class BoundaryPointLossParams(TypedDict):
    bnd_points_size : tuple
    domain_mesh_size : tuple

@dataclass
class Hyperparameters:
    '''
    Defines the configuration of a particular model run. 
    '''
    training_batch_size: int
    test_batch_size: int
    train_excl_boundary_points: bool
    test_excl_boundary_points: bool
    model_cls: Type[torch.nn.Module]
    model_params: dict
    optimizer_cls: Type[torch.optim.Optimizer]
    optimizer_params: dict
    scheduler_cls: object | None
    scheduler_params: dict
    l_weights: bool
    num_epochs: int
    boundary_loss: bool
    # harmonic_u_loss: bool
    harmonic_psi_loss: bool
    train_subset_idx: int = None
    test_subset_idx: int = None
    # harmonic_u_loss_factor: float = 1.0
    harmonic_psi_loss_factor: float = 1.0
    boundary_loss_factor: float = 1.0
    prediction_loss_factor: float = 1.0
    num_runs: int = 1
    device: torch.device = torch.device("mps")
    wandb_project_name: str = "test",
    test_dir: str | None = None,
    multi_mesh_training_variant: multi_mesh_training_method = "standardize"

    def __post_init__(self):
        if self.num_runs < 1:
            raise ValueError("Number of runs in hyperparameter configuration must be at least 1. ")
        
        if type(self.model_cls) == str:
            self.model_cls = self._str_to_cls(self.model_cls)
        
        if type(self.optimizer_cls) == str:
            self.optimizer_cls = self._str_to_cls(self.optimizer_cls)
        
        if type(self.scheduler_cls) == str:
            self.scheduler_cls = self._str_to_cls(self.scheduler_cls)

        
    def _str_to_cls(self, s: str):
        '''
        Converts string of type '<class 'cls_module.cls'>' to the corresponding class.

        Parameters:
            s: string of type '<class 'cls_module.cls'>'
        '''
        full_class_path = s.strip("<class '>").strip("'>") 

        # Split into module and class
        *module_parts, class_name = full_class_path.split(".")
        module_path = ".".join(module_parts)

        # Import module and get class
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls

                
    def get_dict(self):
        output = self.__dict__.copy()
        output["model_cls"] = str(self.model_cls)
        output["optimizer_cls"] = str(self.optimizer_cls)
        if self.scheduler_cls is not None:
            output["scheduler_cls"] = str(self.scheduler_cls)
        output["device"] = str(self.device)
        return output



@dataclass
class ExprParameters:
    u_bnd_expr: sp.Expr
    a_diffusion_expr: sp.Expr
    u_func_exprs: list[sp.Expr]
    f_func_exprs: list[sp.Expr]
    def as_string_dict(self) -> dict:
        return {k: str(v) for k, v in self.__dict__.items()}
@dataclass
class DataGenerationParameters:
    '''
    Defines the parameters used in data generation. Purely for generating a json file for overview.
    '''
    domain: tuple
    evaluation_mesh_size: tuple | list[tuple]
    evaluation_mesh_type: mesh_type
    integration_mesh_size: tuple | list[tuple]
    integration_mesh_type: mesh_type
    params: dict
    diffusion_params: dict

    def get_dict(self):
        output_dict = self.__dict__.copy()
        return output_dict

    
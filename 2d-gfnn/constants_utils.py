from typing import Literal, TypedDict
from dataclasses import dataclass


mesh_type = Literal["uniform", "chebyshev", "random"]
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
    hidden_channels: int
    num_layers: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    step_size: int
    gamma: float
    l_weights: bool
    num_runs: int = 1

    def __post_init__(self):
        if self.num_runs < 1:
            raise ValueError("Number of runs in hyperparameter configuration must be at least 1. ")

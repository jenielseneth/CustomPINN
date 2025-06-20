from typing import Literal, TypedDict
from dataclasses import dataclass


mesh_type = Literal["uniform", "chebyshev", "random"]
class WeightParams(TypedDict):
    domain: tuple
    f_mesh_type: mesh_type
    f_mesh_size: tuple

class BoundaryPointLossParams(TypedDict):
    bnd_points_size : tuple
    domain_mesh_size : tuple

@dataclass
class Hyperparameters:
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
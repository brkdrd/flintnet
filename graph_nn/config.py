from dataclasses import dataclass, field


@dataclass
class Config:
    activation_fn: str = "relu"
    activation_threshold: float = 0.0
    learning_rate: float = 0.01
    max_activations_per_neuron: int = 100
    restructure_interval: int = 50
    high_weight_threshold: float = 1.0
    low_weight_threshold: float = 0.1
    high_gradient_threshold: float = 1.0
    low_gradient_threshold: float = 0.1
    low_activation_threshold: float = 1.0
    low_activation_window: int = 3
    gradient_ema_alpha: float = 0.1
    restructure_top_n: int = 10
    loss_fn: str = "mse"

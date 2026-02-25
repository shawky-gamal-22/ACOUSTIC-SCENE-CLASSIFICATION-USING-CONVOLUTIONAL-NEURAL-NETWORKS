"""Configuration classes for baseline and ablation experiments."""

from dataclasses import dataclass


@dataclass
class BaselineConfig:
    """Baseline configuration."""

    name: str = "baseline"

    # Data
    num_samples: int = 1000
    input_dim: int = 10
    output_dim: int = 2

    # Training
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 0.01

    # Misc
    seed: int = 42
    device: str = "cuda"

    # Audio processing
    target_sample_rate: int = 16000
    n_mels: int = 60
    n_fft: int = int(0.040 * target_sample_rate)
    hop_length: int = n_fft // 2

    # Paths
    annotation_path: str = "data/meta.txt"
    data_path: str = "./data"


@dataclass
class Ablation1Config(BaselineConfig):
    """TODO: First ablation experiment."""

    name: str = "ablation1"
    # TODO: Override parameters for ablation 1
    learning_rate: float = 0.001


@dataclass
class Ablation2Config(BaselineConfig):
    """TODO: Second ablation experiment."""

    name: str = "ablation2"
    # TODO: Override parameters for ablation 2
    batch_size: int = 64


@dataclass
class Ablation3Config(BaselineConfig):
    """TODO: Third ablation experiment."""

    name: str = "ablation3"
    # TODO: Override parameters for ablation 3
    num_epochs: int = 20


def get_config(config_name: str):
    """Get configuration by name."""
    configs = {
        "baseline": BaselineConfig(),
        "ablation1": Ablation1Config(),
        "ablation2": Ablation2Config(),
        "ablation3": Ablation3Config(),
    }
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}")
    return configs[config_name]

"""Configuration classes for baseline and ablation experiments."""

from dataclasses import dataclass


@dataclass
class BaselineConfig:
    """Baseline configuration."""

    name: str = "baseline"

    # Training
    batch_size: int = 128
    non_full_epochs: int = 300
    full_epochs: int = 200
    learning_rate: float = 1e-4
    n_folds: int = 4
    num_classes: int = 15
    eval_every: int = 5
    patience: int = 100

    # Audio processing
    target_sample_rate: int = 44100
    n_mels: int = 60
    n_fft: int = 2048
    hop_length: int = n_fft // 2

    seq_duration: float = 3.0
    audio_length: float = 30.0

    # Paths
    annotation_path: str = "data\\meta.txt"
    data_path: str = ".\\data\\"
    data_folds_path: str = ".\\data\\"
    features_dir: str = ".\\features\\"

    CLASSES = [
        "beach",
        "bus",
        "cafe/restaurant",
        "car",
        "city_center",
        "forest_path",
        "grocery_store",
        "home",
        "library",
        "metro_station",
        "office",
        "park",
        "residential_area",
        "train",
        "tram",
    ]

    model_type: str = "cnn_transformer"
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 3


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

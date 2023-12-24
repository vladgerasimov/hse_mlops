from dataclasses import dataclass


@dataclass
class DatasetConfig:
    train_data_path: str
    val_data_path: str
    test_data_path: str
    train_sample: int
    batch_size: int
    img_height: int
    img_width: int
    img_mean: tuple[float, float, float]
    img_std: tuple[float, float, float]


@dataclass
class ModelTrainingConfig:
    n_epochs: int
    learning_rate: float
    device: str | None
    cpt_path: str


@dataclass
class ModelInferenceConfig:
    pred_path: str


@dataclass
class MLFlowConfig:
    url: str
    experiment_name: str


@dataclass
class AppConfig:
    dataset: DatasetConfig
    model_training: ModelTrainingConfig
    model_inference: ModelInferenceConfig
    mlflow: MLFlowConfig

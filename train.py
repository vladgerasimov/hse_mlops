import hydra
from dvc.api import DVCFileSystem
from loguru import logger
from pathlib import Path

from classification_project.model.cat_dog_cls_model import CatDogClassificationModel
from classification_project.core.data_models import AppConfig


@hydra.main(version_base=None, config_path="configs", config_name="app_config")
def train_model(config: AppConfig):
    fs = DVCFileSystem("./")
    train_data_path = config.dataset.train_data_path
    val_data_path = config.dataset.val_data_path

    logger.info("Downloading data from DVC...")
    if not Path(train_data_path).exists():
        fs.get(train_data_path, train_data_path, recursive=True)
    if not Path(val_data_path).exists():
        fs.get(val_data_path, val_data_path, recursive=True)
    logger.info("Downloaded data from DVC...")

    model = CatDogClassificationModel(
        img_height=config.dataset.img_height,
        img_width=config.dataset.img_width,
        img_mean=config.dataset.img_mean,
        img_std=config.dataset.img_std,
    )

    logger.info("Fitting model...")
    model.fit(
        train_dataset_path=train_data_path,
        val_dataset_path=val_data_path,
        mlflow_config=config.mlflow,
        device=config.model_training.device,
        sample_size=config.dataset.train_sample,
        batch_size=config.dataset.batch_size,
        lr=config.model_training.learning_rate,
        n_epochs=config.model_training.n_epochs,
        cpt_path=config.model_training.cpt_path,
    )
    logger.info("Model fitted successfully")


if __name__ == '__main__':
    train_model()

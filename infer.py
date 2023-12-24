import hydra
from dvc.api import DVCFileSystem
from loguru import logger
from pathlib import Path

from src.model.cat_dog_cls_model import CatDogClassificationModel
from src.core.data_models import AppConfig

fs = DVCFileSystem("./")


@hydra.main(version_base=None, config_path="configs", config_name="app_config")
def infer_model(config: AppConfig):
    cpt_path = config.model_training.cpt_path
    if not Path(cpt_path).exists():
        logger.info("Downloading model checkpoint from DVC...")
        fs.get(cpt_path, cpt_path, recursive=True)
        logger.info("Downloaded model checkpoint from DVC")

    model = CatDogClassificationModel(
        model_path=config.model_training.cpt_path,
        img_height=config.dataset.img_height,
        img_width=config.dataset.img_width,
        img_mean=config.dataset.img_mean,
        img_std=config.dataset.img_std,
    )
    test_dataset_path = config.dataset.test_data_path

    if not Path(test_dataset_path).exists():
        logger.info("Downloading data from DVC...")
        fs.get(test_dataset_path, test_dataset_path, recursive=True)
        logger.info("Downloaded data from DVC...")

    pred_df = model.predict(test_dataset_path, batch_size=config.dataset.batch_size)
    logger.success(pred_df.head())
    pred_df.to_csv(config.model_inference.pred_path)


if __name__ == '__main__':
    infer_model()

from pathlib import Path

import pandas as pd
import torch
import torchvision
from loguru import logger
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from tqdm import tqdm

from src.core.data_models import MLFlowConfig
from src.model.base import TorchModel, get_device, train_model


class CatDogClassificationModel(TorchModel):
    def __init__(
        self,
        model_path: str | Path | None = None,
        img_height: int = 96,
        img_width: int = 96,
        img_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        img_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = torch.nn.Linear(in_features=self.model.fc.in_features, out_features=2)
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_height, img_width)),  # scaling images to fixed size
                transforms.ToTensor(),  # converting to tensors
                transforms.Normalize(img_mean, img_std),  # normalize image data per-channel
            ]
        )
        self.optimizer_state_dict = None
        if model_path:
            checpoint = torch.load(model_path)
            self.model.load_state_dict(checpoint["model_state_dict"])
            self.optimizer_state_dict = checpoint["optimizer_state_dict"]

    def fit(
        self,
        train_dataset_path: str | Path,
        val_dataset_path: str | Path,
        mlflow_config: MLFlowConfig,
        sample_size: int | None = None,
        batch_size: int = 256,
        n_epochs: int = 10,
        device: str | torch.device = get_device(),
        optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
        cpt_path: str | Path = "resnet18.cp",
        lr: float = 1e-3,
        **optimizer_kwargs,
    ) -> torch.nn.Module:
        optimizer = optimizer_class(self.model.parameters(), lr=lr, **optimizer_kwargs)
        if self.optimizer_state_dict:
            optimizer.load_state_dict(self.optimizer_state_dict)

        train_dataset = torchvision.datasets.ImageFolder(train_dataset_path, transform=self.transform)
        logger.info(f"{len(train_dataset)=}")
        val_dataset = torchvision.datasets.ImageFolder(val_dataset_path, transform=self.transform)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=RandomSampler(data_source=train_dataset, num_samples=sample_size) if sample_size else None,
            pin_memory=True,
            num_workers=2,
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=2)

        self.model = train_model(
            self.model,
            optimizer,
            train_loader,
            val_loader,
            n_epochs,
            mlflow_config=mlflow_config,
            cpt_path=cpt_path,
            device=device,
        )

        return self.model

    def predict(self, dataset_path: str | Path, batch_size: int = 256) -> pd.DataFrame:
        dataset = torchvision.datasets.ImageFolder(dataset_path, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        preds = []
        with torch.no_grad():
            self.model.train(False)
            for batch, _ in tqdm(dataloader, desc="Inference", leave=False):
                logits = self.model(batch)
                pred = torch.argmax(logits, dim=1).detach().cpu().tolist()
                preds.extend(pred)
        return pd.DataFrame({"pred": preds})

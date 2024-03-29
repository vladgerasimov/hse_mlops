import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path

import mlflow
import numpy as np
import torch
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from classification_project.core.data_models import MLFlowConfig


def get_commit_hash():
    result = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, text=True, check=True)
    return result.stdout.strip()


class TorchModel(ABC):
    @abstractmethod
    def fit(
        self, train_dataset: Dataset, val_dataset: Dataset, optimizer_class: torch.optim.Optimizer
    ) -> torch.nn.Module:
        """
        The method that fits <self.model> on passed dataset
        :param train_dataset: torch.utils.data.Dataset - dataset for training
        :param val_dataset: torch.utils.data.Dataset - dataset for validation
        :param optimizer_class: torch.optim.Optimizer - optimizer
        :return: torch.nn.Module - torch model object
        """
        ...

    @abstractmethod
    def predict(self, batch: torch.tensor) -> torch.tensor:
        """
        Inferences model on passed objects
        :param batch: torch.tensor - objects for prediction
        :return: torch.tensor - predictions
        """
        ...


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def calculate_metrics(pred, target) -> dict[str, float]:
    metrics = {"accuracy": accuracy_score, "f1_score": f1_score}

    result = {}
    for metric_name, metric in metrics.items():
        result[metric_name] = metric(target, pred)

    return result


def compute_loss(model, data_batch, loss_function: torch.nn.modules.loss._WeightedLoss):
    """Compute the loss using loss_function for the batch of data and return mean loss value for this batch."""
    # load the data
    img_batch = data_batch["img"]
    label_batch = data_batch["label"]
    # forward pass
    logits = model(img_batch)

    # loss computation
    loss = loss_function(logits, label_batch)

    return loss


@torch.no_grad()  # we do not need to save gradients on evaluation
def test_model(
    model,
    batch_generator,
    epoch: int,
    loss_function: torch.nn.modules.loss._WeightedLoss,
    device: torch.device = get_device(),
):
    """Evaluate the model using data from batch_generator and metrics defined above."""
    # disable dropout / use averages for batch_norm
    model.train(False)

    # save loss values for performance logging
    loss_list = []
    pred = []
    target = []

    for X_batch, y_batch in tqdm(batch_generator, desc="Validation", leave=False):
        # do the forward pass
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(X_batch)

        # compute loss value
        loss = loss_function(logits, y_batch)

        # save the necessary data
        loss_list.append(loss.detach().cpu().numpy().tolist())
        labels = logits.argmax(dim=1).cpu().tolist()
        pred.extend(labels)
        target.extend(y_batch.cpu().tolist())

    metrics = calculate_metrics(pred, target)
    mlflow.log_metrics(metrics, step=epoch)

    return loss_list


def train_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int,
    mlflow_config: MLFlowConfig,
    device: torch.device | str = get_device(),
    cpt_path: str | Path | None = None,
    loss_function: torch.nn.modules.loss._WeightedLoss = torch.nn.CrossEntropyLoss(),
) -> torch.nn.Module:
    mlflow.set_tracking_uri(mlflow_config.url)
    if not mlflow.get_experiment_by_name(mlflow_config.experiment_name):
        mlflow.create_experiment(mlflow_config.experiment_name)
    mlflow.set_experiment(mlflow_config.experiment_name)

    if isinstance(device, str):
        device = torch.device(device)
    elif device is None:
        device = get_device()
    logger.info("Start training of model with {} epochs", n_epochs)
    logger.info(f"Using {device=}")
    model = model.to(device)

    train_loss, val_loss = [], [1]
    val_loss_idx = [0]
    least_loss_val = np.inf

    with mlflow.start_run():
        params = {
            "learning_rate": optimizer.param_groups[0]["lr"],
            "n_epochs": n_epochs,
            "batch_size": train_loader.batch_size,
            "cpt_path": cpt_path,
            "git_commit_hash": get_commit_hash(),
        }
        mlflow.log_params(params)

        for epoch in range(n_epochs):
            start_time = time.time()

            # Train phase
            model.train(True)  # enable dropout / batch_norm training behavior
            for X_batch, y_batch in tqdm(train_loader, desc="Training", leave=False):
                # move data to target device
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                data_batch = {"img": X_batch, "label": y_batch}

                # train on batch: compute loss, calc grads, perform optimizer step and zero the grads
                loss = compute_loss(model, data_batch, loss_function)

                # compute backward pass
                loss.backward()
                optimizer.step()

                # log train loss
                train_loss.append(loss.detach().cpu().numpy())

            mean_train_loss = np.array(train_loss).mean()

            # Evaluation phase
            losses = test_model(model, val_loader, epoch, loss_function, device)

            # Logging
            val_loss_value = np.mean(losses)
            val_loss_idx.append(len(train_loss))
            val_loss.append(val_loss_value)

            mlflow.log_metric("train_loss", value=mean_train_loss, step=epoch)
            mlflow.log_metric("val_loss", value=val_loss_value, step=epoch)

            logger.info("Epoch {} of {} took {:.3f}s", epoch + 1, n_epochs, time.time() - start_time)
            if val_loss_value < least_loss_val and cpt_path is not None:
                logger.success(
                    f"Best validation loss found: {val_loss_value} < {least_loss_val};"
                    f"saving checkpoint to '{cpt_path}'"
                )
                least_loss_val = val_loss_value
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    cpt_path,
                )

    return model

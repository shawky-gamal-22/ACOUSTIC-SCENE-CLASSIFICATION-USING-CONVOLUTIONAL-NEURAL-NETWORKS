"""Training script."""

import argparse
import torch
import torch.optim as optim
import wandb

from config import get_config
from model import create_model
from data import get_dataloaders

from loss import get_loss_fn
from metrics import compute_metrics
from utils import set_seed, get_device
from evaluate import evaluate
from tqdm import tqdm


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for data, target in tqdm(dataloader, desc="Training"):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return {
        "loss": total_loss / len(dataloader),
    }


def non_full_training(
    model,
    data_loader,
    validation_data,
    criterion,
    optimizer,
    config,
    device,
    fold_number,
):

    patience_counter = 0
    best_val_accuracy = 0.0
    patience = 100
    best_epoch = 0

    for epoch in range(config.non_full_epochs):
        train_loss = train_epoch(model, data_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, validation_data, config, device)

        if epoch % config.eval_every == 0:
            if val_metrics > best_val_accuracy:
                best_val_accuracy = val_metrics
                patience_counter = 0
                best_epoch = epoch
            else:
                patience_counter += config.eval_every
            if patience_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch + 1}. Best validation accuracy: {best_val_accuracy:.4f} at epoch {best_epoch + 1}"
                )
                break

        print(
            f"Fold {fold_number}: Epoch {epoch + 1}: Train Loss: {train_loss['loss']:.4f}, Val Metrics: {val_metrics:.4f}"
        )

    return best_epoch


def full_training(
    model,
    data_loader,
    val_data,
    criterion,
    optimizer,
    config,
    device,
    best_epoch,
    fold_number,
):
    for epoch in range(best_epoch):
        train_loss = train_epoch(model, data_loader, criterion, optimizer, device)

        if epoch % config.eval_every == 0:
            val_metrics = evaluate(model, val_data, config, device)

        if epoch % 10 == 0:
            print(
                f"Fold {fold_number}: Epoch {epoch + 1}: Train Loss: {train_loss['loss']:.4f}, Val Metrics: {val_metrics}"
            )
    return model


def train(config):

    for fold in range(1, config.n_folds + 1):
        model = create_model(config)
        device = get_device("cuda")
        model.to(device)
        criterion = get_loss_fn()
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        non_full_training_dataloader, non_full_validation_data = get_dataloaders(
            config=config, fold=fold, state="Non_full_training"
        )

        best_epoch = non_full_training(
            model,
            non_full_training_dataloader,
            non_full_validation_data,
            criterion,
            optimizer,
            config,
            device,
            fold,
        )
        model = create_model(config)
        device = get_device("cuda")
        model.to(device)
        criterion = get_loss_fn()
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        full_training_dataloader, full_training_val_data = get_dataloaders(
            config=config, fold=fold, state="Full_training"
        )

        model = full_training(
            model,
            full_training_dataloader,
            full_training_val_data,
            criterion,
            optimizer,
            config,
            device,
            best_epoch,
            fold,
        )

        fold_metrics = evaluate(model, full_training_val_data, criterion, device)
        print(f"Fold {fold} Metrics: {fold_metrics}")


def main():
    config = get_config("baseline")
    train(config)


if __name__ == "__main__":

    main()

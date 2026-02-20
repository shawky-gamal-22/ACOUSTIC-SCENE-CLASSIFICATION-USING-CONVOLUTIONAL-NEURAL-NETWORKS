"""Training script."""

import argparse
import torch
import torch.optim as optim
import wandb

from config import get_config
from model import create_model
from data import get_dataloader
from loss import get_loss_fn
from metrics import compute_metrics
from utils import set_seed, get_device
from evaluate import evaluate


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_acc = 0
    
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        metrics = compute_metrics(output, target)
        total_acc += metrics['accuracy']
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': total_acc / len(dataloader)
    }


def train(config):
    """Main training function."""
    # Initialize wandb
    wandb.init(project="dl-course", name=config.name, config=vars(config))
    
    set_seed(config.seed)
    device = get_device(config.device)
    
    # Setup
    train_loader = get_dataloader(config, 'train')
    val_loader = get_dataloader(config, 'val')
    model = create_model(config).to(device)
    criterion = get_loss_fn()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    for epoch in range(config.num_epochs):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train/loss': train_metrics['loss'],
            'train/accuracy': train_metrics['accuracy'],
            'val/loss': val_metrics['loss'],
            'val/accuracy': val_metrics['accuracy']
        })
        
        print(f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, Val Acc={val_metrics['accuracy']:.4f}")
    
    wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, 
                       choices=['baseline', 'ablation1', 'ablation2', 'ablation3'],
                       help='Configuration to use')
    args = parser.parse_args()
    
    config = get_config(args.config)
    train(config)


if __name__ == "__main__":
    main()

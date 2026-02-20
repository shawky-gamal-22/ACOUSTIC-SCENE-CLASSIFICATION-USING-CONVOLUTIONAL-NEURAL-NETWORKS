"""Evaluation script."""

import argparse
import torch

from config import get_config
from model import create_model
from data import get_dataloader
from loss import get_loss_fn
from metrics import compute_metrics
from utils import set_seed, get_device


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_acc = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            metrics = compute_metrics(output, target)
            total_acc += metrics['accuracy']
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': total_acc / len(dataloader)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    
    config = get_config(args.config)
    set_seed(config.seed)
    device = get_device(config.device)
    
    test_loader = get_dataloader(config, 'test')
    model = create_model(config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    criterion = get_loss_fn()
    results = evaluate(model, test_loader, criterion, device)
    
    print(f"Test Loss: {results['loss']:.4f}")
    print(f"Test Accuracy: {results['accuracy']:.4f}")


if __name__ == "__main__":
    main()

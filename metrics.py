"""Metrics computation."""

import torch


def compute_accuracy(output, target):
    """Compute classification accuracy."""
    pred = output.argmax(dim=1)
    correct = (pred == target).sum().item()
    total = target.size(0)
    return correct / total


def compute_metrics(output, target):
    """
    Compute all metrics.
    
    TODO: Add your custom metrics here.
    """
    return {
        'accuracy': compute_accuracy(output, target)
    }

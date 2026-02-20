"""Data loading."""

import torch
from torch.utils.data import Dataset, DataLoader


class SimpleDataset(Dataset):
    """
    Simple dataset with random data and deterministic labels.
    
    TODO: Replace with your actual dataset.
    """
    
    def __init__(self, num_samples, input_dim, seed):
        self.num_samples = num_samples
        self.input_dim = input_dim
        
        # Generate random data
        torch.manual_seed(seed)
        self.X = torch.randn(num_samples, input_dim)
        
        # Deterministic labels based on sum of features
        self.y = (self.X.sum(dim=1) > 0).long()
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloader(config, split='train'):
    """Create dataloader."""
    seed = config.seed if split == 'train' else config.seed + 1
    
    dataset = SimpleDataset(
        num_samples=config.num_samples,
        input_dim=config.input_dim,
        seed=seed
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == 'train')
    )
    
    return dataloader

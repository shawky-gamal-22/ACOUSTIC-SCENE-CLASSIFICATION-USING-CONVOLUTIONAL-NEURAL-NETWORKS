# Deep Learning Course Project

Minimal template for implementing papers with PyTorch.

## Structure

```
.
├── config.py      # Baseline + 3 ablation configs
├── model.py       # Model architecture
├── data.py        # Data loading
├── loss.py        # Loss functions
├── metrics.py     # Metrics computation
├── utils.py       # Utilities
├── train.py       # Training script
└── evaluate.py    # Evaluation script
```

## Setup

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create Environment

```bash
uv venv --python 3.11
source .venv/bin/activate  # Linux/macOS
```

### Install Dependencies

```bash
uv pip install -r requirements.txt
```

### Login to Weights & Biases

```bash
wandb login
```

## Usage

### Training

Run baseline:
```bash
python train.py --config baseline
```

Run ablation experiments:
```bash
python train.py --config ablation1
python train.py --config ablation2
python train.py --config ablation3
```

### Evaluation

```bash
python evaluate.py --config baseline --checkpoint path/to/checkpoint.pth
```

## Implementation Guide

1. **config.py**: Define parameters for each ablation
2. **model.py**: Implement your architecture
3. **data.py**: Replace with your dataset
4. **loss.py**: Define custom loss if needed
5. **metrics.py**: Add evaluation metrics
6. **train.py**: Modify training loop if needed

## Ablation Studies

Each ablation config inherits from `BaselineConfig`. Override specific parameters to test:
- Different hyperparameters
- Architectural changes
- Training strategies

Results are tracked automatically in Weights & Biases.

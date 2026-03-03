"""Evaluation script."""

import argparse
import torch

from config import get_config
from model import create_model
from data import get_dataloaders
from loss import get_loss_fn
from utils import set_seed, get_device
import numpy as np


def evaluate(model, data, config, device):
    """Evaluate model."""

    model.eval()
    file_ids, labels, normalized_features = data

    frames_per_seq = int(
        config.seq_duration * config.target_sample_rate / config.hop_length
    )
    target_w = 130

    class_correct = {c: 0 for c in range(len(config.CLASSES))}
    class_total = {c: 0 for c in range(len(config.CLASSES))}

    with torch.no_grad():
        for file_id, label in zip(file_ids, labels):
            log_mel = normalized_features[file_id]
            total_T = log_mel.shape[2]
            num_seqs = total_T // frames_per_seq

            if num_seqs == 0:
                continue

            all_seqs = []

            for i in range(num_seqs):
                start = i * frames_per_seq
                end = start + frames_per_seq

                seq = log_mel[:, :, start:end]

                curr_w = seq.shape[2]

                if curr_w < target_w:
                    pad_width = target_w - curr_w
                    seq = np.pad(
                        seq,
                        ((0, 0), (0, 0), (0, pad_width)),
                        mode="constant",
                        constant_values=0,
                    )
                elif curr_w > target_w:
                    seq = seq[:, :, :target_w]
                all_seqs.append(seq)

            seqs_t = torch.tensor(np.stack(all_seqs, axis=0), dtype=torch.float32).to(
                device
            )
            logits = model(seqs_t)
            probs = torch.softmax(logits, dim=1)
            avg_probs = probs.mean(dim=0)
            pred_label = avg_probs.argmax().item()

            if pred_label == label:
                class_correct[label] += 1
            class_total[label] += 1

        per_class = [
            class_correct[c] / class_total[c] if class_total[c] > 0 else 0
            for c in range(len(config.CLASSES))
        ]
        return float(np.mean(per_class))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    config = get_config(args.config)
    set_seed(config.seed)
    device = get_device(config.device)

    test_loader = get_dataloaders(config, "test")
    model = create_model(config).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = get_loss_fn()
    results = evaluate(model, test_loader, criterion, device)

    print(f"Test Loss: {results['loss']:.4f}")
    print(f"Test Accuracy: {results['accuracy']:.4f}")


if __name__ == "__main__":
    main()

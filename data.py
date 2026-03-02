"""Data loading."""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
import torchaudio.transforms as T
import os
from config import BaselineConfig
import glob
import numpy as np
from pathlib import Path


def load_fold_data(data_dir: str):
    config = BaselineConfig()
    label_map = {cls: idx for idx, cls in enumerate(config.CLASSES)}

    fold_data = {}

    for k in range(1, 5):
        fold_data[k] = {"train": [], "test": []}

        for split, key in [("train", "train"), ("evaluation", "test")]:
            fpath = os.path.join(data_dir, "evaluation_setup", f"fold{k}_{split}.txt")
            with open(fpath, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")

                    if len(parts) <= 2:
                        continue
                    file_id = Path(parts[0]).stem
                    label = parts[1].strip()
                    fold_data[k][key].append((file_id, label_map[label]))

    return fold_data, label_map


def extract_mel_spectrogram(audio_path: str, config: BaselineConfig):
    signal, sample_rate = torchaudio.load(audio_path)

    if sample_rate != config.target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=config.target_sample_rate
        )
        signal = resampler(signal)

    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)

    transformation = torch.nn.Sequential(
        T.MelSpectrogram(
            sample_rate=config.target_sample_rate,
            n_fft=config.n_fft,
            win_length=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            norm="slaney",
        ),
        T.AmplitudeToDB(),
    )

    mel_spectrogram = transformation(signal)

    return mel_spectrogram


def precompute_all_features(config: BaselineConfig):

    os.makedirs(config.features_dir, exist_ok=True)
    features = {}

    audio_files = glob.glob(os.path.join(config.data_path, "audio", "*.wav"))

    for audio_file in audio_files:
        file_id = os.path.splitext(os.path.basename(audio_file))[0]
        feat_path = os.path.join(config.features_dir, f"{file_id}.npy")

        if os.path.exists(feat_path):
            features[file_id] = np.load(feat_path)
        else:
            log_mel = extract_mel_spectrogram(audio_file, config)
            np.save(feat_path, log_mel.numpy())
            features[file_id] = log_mel.numpy()
    return features


def normalize_per_fold(features: dict, train_ids: list) -> tuple:
    """Paper: 'normalize each bin by subtracting its mean and dividing by its
    std, both calculated on the whole training set of each fold.'"""

    train_specs = np.concatenate([features[file_id] for file_id in train_ids], axis=1)
    mean = np.mean(train_specs, axis=1, keepdims=True)
    std = np.std(train_specs, axis=1, keepdims=True)
    std = np.where(std == 0, 1e-8, std)  # Avoid division by zero

    normalized_features = {}
    for file_id in features:
        normalized_features[file_id] = (features[file_id] - mean) / std

    return normalized_features, mean, std


class AudioDataset(Dataset):
    """
    Simple dataset with random data and deterministic labels.

    TODO: Replace with your actual dataset.
    """

    def __init__(
        self, file_ids, labels, features, config: BaselineConfig, augment: bool = False
    ):
        self.file_ids = file_ids
        self.labels = labels
        self.features = features
        self.config = config
        self.augment = augment

        self.frames_per_seq = int(
            self.config.seq_duration
            * self.config.target_sample_rate
            / self.config.hop_length
        )

        self.items = []

        for file_id, label in zip(file_ids, labels):
            log_mel = self.features[file_id]
            total_T = log_mel.shape[1]

            if augment:
                max_shift = self.frames_per_seq // 4
                shift = np.random.randint(-max_shift, max_shift)
                log_mel = np.roll(log_mel, shift, axis=1)

            num_seqs = total_T // self.frames_per_seq

            all_seq_per_audio = []
            for seq_idx in range(num_seqs):
                start = seq_idx * self.frames_per_seq
                end = start + self.frames_per_seq
                seq_log_mel = log_mel[:, start:end]  # Shape: (n_mels, frames_per_seq)
                all_seq_per_audio.append(seq_log_mel)
                # self.items.append(
                #     (seq_log_mel[np.newaxis, ...], label)
                # )  # Add channel dimension (1, n_mels, frames_per_seq)
            self.items.append(
                (np.stack(all_seq_per_audio, axis=0), label)
            )  # Shape: (num_seqs, n_mels, frames_per_seq)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        all_seqs, label = self.items[idx]
        return (
            torch.tensor(all_seqs, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )

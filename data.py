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

        for split, key in [("train", "train"), ("evaluate", "test")]:
            fpath = os.path.join(data_dir, "evaluation_setup", f"fold{k}_{split}.txt")
            with open(fpath, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")

                    if len(parts) < 2:
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

    "normalize the train_ids using the same mean and std"
    for file_id in features.keys():

        normalized_features[file_id] = (features[file_id] - mean) / std

    return normalized_features, mean, std


def get_dataloaders(config, fold, state: str):

    fold_data, _ = load_fold_data("./")
    features = precompute_all_features(config)

    train_ids, train_labels = zip(*fold_data[fold]["train"])
    # split 80 ,20
    test_ids, test_labels = zip(*fold_data[fold]["test"])

    normalized_features, _, _ = normalize_per_fold(features, train_ids)

    if state == "Non_full_tarining":

        # slice with stratified sampling to maintain class distribution
        unique_labels = list(set(train_labels))
        train_indices = []
        val_indices = []

        for label in unique_labels:
            label_indices = [i for i, l in enumerate(train_labels) if l == label]
            np.random.shuffle(label_indices)

            split_point = int(0.8 * len(label_indices))
            train_indices.extend(label_indices[:split_point])
            val_indices.extend(label_indices[split_point:])

        train_ids = [train_ids[i] for i in train_indices]
        train_labels = [train_labels[i] for i in train_indices]

        validation_ids = [train_ids[i] for i in val_indices]
        validation_labels = [train_labels[i] for i in val_indices]

        train_dataset = AudioDataset(
            train_ids, train_labels, normalized_features, config, augment=True
        )
        validation_dataset = AudioDataset(
            validation_ids,
            validation_labels,
            normalized_features,
            config,
            augment=False,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size=config.batch_size, shuffle=False
        )
        return train_loader, validation_loader

    else:

        train_dataset = AudioDataset(
            train_ids, train_labels, normalized_features, config, augment=True
        )

        test_dataset = AudioDataset(
            test_ids, test_labels, normalized_features, config, augment=False
        )

        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config.batch_size, shuffle=False
        )

        return train_loader, test_loader


class AudioDataset(Dataset):
    def __init__(self, file_ids, labels, features, config, augment: bool = False):
        self.config = config
        self.augment = augment
        self.items = []

        self.frames_per_seq = int(
            self.config.seq_duration
            * self.config.target_sample_rate
            / self.config.hop_length
        )

        for file_id, label in zip(file_ids, labels):
            log_mel = features[file_id]  # (60, T)
            total_T = log_mel.shape[1]
            num_seqs = total_T // self.frames_per_seq

            all_seq_per_audio = []

            for seq_idx in range(num_seqs):
                start = seq_idx * self.frames_per_seq

                if self.augment:
                    max_shift = self.frames_per_seq // 4
                    shift = np.random.randint(-max_shift, max_shift)
                    start = max(0, start + shift)

                end = start + self.frames_per_seq

                if end > total_T:
                    end = total_T
                    start = end - self.frames_per_seq

                seq = log_mel[:, :, start:end]  # (60, 150)

                all_seq_per_audio.append(seq)

            # تجميع الـ 10 مقاطع في مصفوفة واحدة (10, 1, 60, 150)
            # الـ stack بيحول قائمة المصفوفات لمصفوفة واحدة بأبعاد جديدة
            all_seq_tensor = np.stack(all_seq_per_audio, axis=0)

            self.items.append((all_seq_tensor, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        all_seqs, label = self.items[idx]
        return (
            torch.tensor(all_seqs, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


if __name__ == "__main__":

    config = BaselineConfig()
    fold_data, label_map = load_fold_data(config.data_folds_path)

    features = precompute_all_features(config)

    fold_one = fold_data[1]["train"][0]

    print(f"Example file ID: {fold_one[0]}, label: {fold_one[1]}")

"""Data loading."""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
import torchaudio.transforms as T
import os
from config import BaselineConfig


class AudioDataset(Dataset):
    """
    Simple dataset with random data and deterministic labels.

    TODO: Replace with your actual dataset.
    """

    def __init__(self, annotation_path, data_path, transformation, target_sample_rate):
        self.annotation = pd.read_csv(
            annotation_path, sep="\t", header=None, names=["file_path", "label"]
        )
        self.data_path = data_path
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        audio_sample_path = self._get_sample_path(idx)
        audio_sample_label = self._get_sample_label(idx)
        # audio_sample_path = "D:\\ITI\\projects\\DeepLearning\\ACOUSTIC-SCENE-CLASSIFICATION-USING-CONVOLUTIONAL-NEURAL-NETWORKS\\data\\audio\\a062_120_150.wav"
        if not os.path.exists(audio_sample_path):
            raise FileNotFoundError(f"Audio file not found: {audio_sample_path}")

        signal, sample_rate = torchaudio.load(audio_sample_path)

        signal = self._resample_if_necessary(signal, sample_rate)
        signal = self._mix_down_if_necessary(signal)

        if self.transformation:
            signal = self.transformation(signal)

        return signal, audio_sample_label

    def _resample_if_necessary(self, signal, sample_rate):
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.target_sample_rate
            )
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_sample_path(self, idx):
        file_name = self.annotation.iloc[idx]["file_path"].strip()
        full_path = os.path.join(self.data_path, file_name)
        audio_sample_path = os.path.abspath(full_path)
        return audio_sample_path

    def _get_sample_label(self, idx):
        return self.annotation.iloc[idx]["label"]


def get_dataloader(config, split="train"):
    """Create dataloader."""
    seed = config.seed if split == "train" else config.seed + 1

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

    dataset = AudioDataset(
        annotation_path=config.annotation_path,
        data_path=config.data_path,
        transformation=transformation,
        target_sample_rate=config.target_sample_rate,
    )

    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=(split == "train")
    )

    return dataloader


def __main__():
    """Test SimpleDataset."""

    config = BaselineConfig()

    data_loader = get_dataloader(config, split="train")

    signal, audio_sample_label = next(iter(data_loader))
    print(f"Signal shape: {signal.shape}")
    print(f"Audio sample label: {audio_sample_label}")


if __name__ == "__main__":
    __main__()

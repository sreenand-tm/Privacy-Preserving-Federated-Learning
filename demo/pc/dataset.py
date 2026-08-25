"""
PyTorch Dataset for loading preprocessed mel-spectrogram features.
Supports fold-based train/test splitting for 10-fold CV.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from config import METADATA_CSV, FEATURES_DIR, BATCH_SIZE


class UrbanSoundDataset(Dataset):
    """Dataset that loads precomputed mel-spectrogram .npy files."""

    def __init__(self, metadata, features_dir=FEATURES_DIR):
        """
        Args:
            metadata: DataFrame with columns [slice_file_name, fold, classID, class]
            features_dir: path to preprocessed features
        """
        self.metadata = metadata.reset_index(drop=True)
        self.features_dir = features_dir

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_name = row['slice_file_name']
        fold = row['fold']
        class_id = row['classID']

        # Load precomputed mel-spectrogram
        npy_name = os.path.splitext(file_name)[0] + ".npy"
        npy_path = os.path.join(self.features_dir, f"fold{fold}", npy_name)
        mel_spec = np.load(npy_path)

        # Convert to tensor: add channel dimension → (1, n_mels, time_frames)
        mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)

        return mel_tensor, class_id


def get_fold_dataloaders(test_fold, batch_size=BATCH_SIZE):
    """
    Create train and test DataLoaders for a given test fold.
    Train = all folds except test_fold.
    Test = test_fold only.

    This follows UrbanSound8K's mandated 10-fold CV protocol.
    """
    metadata = pd.read_csv(METADATA_CSV)

    train_meta = metadata[metadata['fold'] != test_fold]
    test_meta = metadata[metadata['fold'] == test_fold]

    train_dataset = UrbanSoundDataset(train_meta)
    test_dataset = UrbanSoundDataset(test_meta)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )

    print(f"Fold {test_fold} | Train: {len(train_dataset)} | Test: {len(test_dataset)}")
    return train_loader, test_loader

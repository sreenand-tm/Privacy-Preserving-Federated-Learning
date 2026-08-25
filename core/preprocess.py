"""
Feature extraction: converts raw audio clips to mel-spectrograms and saves them
as .npy files for fast loading during training.

Uses soundfile + scipy + numpy only (no librosa/torchaudio/numba).

Run once:  python preprocess.py
"""
import os
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from scipy.fft import rfft
from math import gcd
from tqdm import tqdm
from config import (
    METADATA_CSV, AUDIO_DIR, FEATURES_DIR,
    SAMPLE_RATE, AUDIO_DURATION, N_MELS, N_FFT, HOP_LENGTH
)


def hz_to_mel(hz):
    """Convert Hz to Mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    """Convert Mel scale to Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    """Create a mel filterbank matrix."""
    if fmax is None:
        fmax = sr / 2.0

    # Mel points
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # FFT bin frequencies
    freq_bins = np.linspace(0, sr / 2.0, n_fft // 2 + 1)

    # Build filterbank
    filterbank = np.zeros((n_mels, n_fft // 2 + 1))

    for i in range(n_mels):
        lower = hz_points[i]
        center = hz_points[i + 1]
        upper = hz_points[i + 2]

        # Rising slope
        for j, freq in enumerate(freq_bins):
            if lower <= freq <= center and center != lower:
                filterbank[i, j] = (freq - lower) / (center - lower)
            elif center < freq <= upper and upper != center:
                filterbank[i, j] = (upper - freq) / (upper - center)

    return filterbank


# Pre-compute mel filterbank (done once)
MEL_FILTERBANK = create_mel_filterbank(SAMPLE_RATE, N_FFT, N_MELS)


def compute_mel_spectrogram(y, sr):
    """Compute log-mel spectrogram from waveform using scipy/numpy."""
    # Window
    window = np.hanning(N_FFT)

    # Number of frames
    n_samples = len(y)
    n_frames = 1 + (n_samples - N_FFT) // HOP_LENGTH

    # STFT
    stft_matrix = np.zeros((N_FFT // 2 + 1, n_frames))
    for i in range(n_frames):
        start = i * HOP_LENGTH
        frame = y[start:start + N_FFT] * window
        spectrum = rfft(frame)
        stft_matrix[:, i] = np.abs(spectrum) ** 2  # power spectrogram

    # Apply mel filterbank
    mel_spec = MEL_FILTERBANK @ stft_matrix

    # Avoid log(0)
    mel_spec = np.maximum(mel_spec, 1e-10)

    # Convert to dB
    log_mel_spec = 10.0 * np.log10(mel_spec)

    # Normalize: set max to 0 dB, clip at -80 dB
    log_mel_spec = log_mel_spec - np.max(log_mel_spec)
    log_mel_spec = np.maximum(log_mel_spec, -80.0)

    return log_mel_spec


def load_and_resample(file_path, target_sr):
    """Load audio file and resample to target sample rate."""
    y, sr = sf.read(file_path, dtype='float32')

    # Convert to mono if stereo
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Resample if needed
    if sr != target_sr:
        g = gcd(int(sr), int(target_sr))
        up = int(target_sr) // g
        down = int(sr) // g
        y = resample_poly(y, up, down).astype(np.float32)

    return y


def extract_mel_spectrogram(file_path):
    """Load audio, resample, pad/truncate, extract log-mel spectrogram."""
    y = load_and_resample(file_path, SAMPLE_RATE)

    # Pad or truncate to fixed length
    target_length = int(SAMPLE_RATE * AUDIO_DURATION)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    else:
        y = y[:target_length]

    # Compute mel spectrogram
    log_mel_spec = compute_mel_spectrogram(y, SAMPLE_RATE)

    return log_mel_spec


def preprocess_dataset():
    """Extract mel-spectrograms for all clips and save as .npy files."""
    os.makedirs(FEATURES_DIR, exist_ok=True)

    # Load metadata
    metadata = pd.read_csv(METADATA_CSV)
    print(f"Total clips: {len(metadata)}")
    print(f"Classes: {metadata['class'].unique()}")
    print(f"Folds: {sorted(metadata['fold'].unique())}")

    success_count = 0
    fail_count = 0

    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Extracting features"):
        file_name = row['slice_file_name']
        fold = row['fold']
        class_id = row['classID']

        audio_path = os.path.join(AUDIO_DIR, f"fold{fold}", file_name)

        if not os.path.exists(audio_path):
            print(f"  [SKIP] File not found: {audio_path}")
            fail_count += 1
            continue

        try:
            mel_spec = extract_mel_spectrogram(audio_path)

            # Save as: features/fold{N}/{filename}.npy
            fold_dir = os.path.join(FEATURES_DIR, f"fold{fold}")
            os.makedirs(fold_dir, exist_ok=True)

            save_name = os.path.splitext(file_name)[0] + ".npy"
            np.save(os.path.join(fold_dir, save_name), mel_spec)

            success_count += 1
        except Exception as e:
            print(f"  [ERROR] {file_name}: {e}")
            fail_count += 1

    print(f"\nDone! Processed: {success_count}, Failed: {fail_count}")
    print(f"Features saved to: {FEATURES_DIR}")

    # Verify shape of one sample
    sample_files = [f for f in os.listdir(os.path.join(FEATURES_DIR, "fold1")) if f.endswith('.npy')]
    if sample_files:
        sample = np.load(os.path.join(FEATURES_DIR, "fold1", sample_files[0]))
        print(f"Sample spectrogram shape: {sample.shape}")  # Expected: (128, ~173)


if __name__ == "__main__":
    preprocess_dataset()

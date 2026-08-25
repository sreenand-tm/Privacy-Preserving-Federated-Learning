"""
Central configuration for the Federated Urban Sound Classification project.
All paths, hyperparameters, and constants in one place.
"""
import os

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(PROJECT_ROOT, "UrbanSound8K", "UrbanSound8K")
AUDIO_DIR = os.path.join(DATASET_ROOT, "audio")
METADATA_CSV = os.path.join(DATASET_ROOT, "metadata", "UrbanSound8K.csv")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "features")  # preprocessed mel-spectrograms

# ─── Audio / Feature Extraction ─────────────────────────────────────────────
SAMPLE_RATE = 22050          # resample all clips to this rate
AUDIO_DURATION = 4.0         # seconds — pad or truncate to this length
N_MELS = 128                 # number of mel bands
N_FFT = 2048                 # FFT window size
HOP_LENGTH = 512             # hop between FFT windows
# With SR=22050, duration=4s, hop=512: spectrogram width = (22050*4)/512 ≈ 172 frames

# ─── Model ───────────────────────────────────────────────────────────────────
NUM_CLASSES = 10
CLASS_NAMES = [
    "air_conditioner",  # 0
    "car_horn",         # 1
    "children_playing", # 2
    "dog_bark",         # 3
    "drilling",         # 4
    "engine_idling",    # 5
    "gun_shot",         # 6
    "jackhammer",       # 7
    "siren",            # 8
    "street_music",     # 9
]

# ─── Training ────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-4
NUM_FOLDS = 10               # UrbanSound8K mandates 10-fold CV

# ─── Device ──────────────────────────────────────────────────────────────────
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

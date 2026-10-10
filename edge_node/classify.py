"""
classify.py — Core Audio Classification Engine for Raspberry Pi
================================================================
Captures audio, extracts Mel-spectrogram features, and classifies
urban sounds using the MobileNetV2 ONNX model.

Supports both:
  - Live microphone input (Bluetooth mic via PipeWire's pw-record)
  - Pre-recorded WAV file input (for demo without hardware mic)
"""

import os
import numpy as np
import time

TARGET_SR    = 22050
DURATION     = 4.0
N_MELS       = 128
N_FFT        = 2048
HOP_LENGTH   = 512

# PipeWire target for the Bluetooth mic (from `wpctl status` -> Sources)
MIC_TARGET_ID = "85"

CLASSES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark",
    "drilling", "engine_idling", "gun_shot", "jackhammer",
    "siren", "street_music", "background_noise"
]

CLASS_LABELS = {
    "air_conditioner":  "Air Conditioner",
    "car_horn":         "Car Horn",
    "children_playing": "Children Playing",
    "dog_bark":         "Dog Bark",
    "drilling":         "Drilling",
    "engine_idling":    "Engine Idling",
    "gun_shot":         "Gun Shot",
    "jackhammer":       "Jackhammer",
    "siren":            "Siren",
    "street_music":     "Street Music",
    "background_noise": "Background Noise"
}

CLASS_ICONS = {
    "air_conditioner":  "❄️", "car_horn":         "📯",
    "children_playing": "🧒", "dog_bark":         "🐕",
    "drilling":         "🔧", "engine_idling":    "🚗",
    "gun_shot":         "💥", "jackhammer":       "⚒️",
    "siren":            "🚨", "street_music":     "🎸",
    "background_noise": "🤫"
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "model_v3_robust_yamnet.onnx")


def compute_mel(audio):
    """Compute normalized log-mel spectrogram matching PyTorch training pipeline."""
    import librosa

    target_len = int(TARGET_SR * DURATION)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    mel = librosa.feature.melspectrogram(
        y=audio, sr=TARGET_SR,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=0.0, fmax=TARGET_SR / 2.0
    )
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    norm_mel = (log_mel + 80.0) / 80.0
    return norm_mel.astype(np.float32)


class EdgeClassifier:
    """ONNX-based urban sound classifier for edge deployment."""

    def __init__(self, model_path=None):
        import onnxruntime as ort

        if model_path is None:
            model_path = DEFAULT_MODEL_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at: {model_path}\n"
                f"Please copy model_v3_robust_yamnet.onnx into {SCRIPT_DIR}"
            )

        self.session = ort.InferenceSession(model_path)
        self.model_name = os.path.basename(model_path)
        print(f"  [Classifier] Loaded: {self.model_name}")

    def predict(self, audio):
        """Classify audio array. Returns (class_name, confidence, label, icon)."""
        mel = compute_mel(audio)
        tensor = np.expand_dims(np.expand_dims(mel, axis=0), axis=0)  # [1, 1, 128, T]

        out = self.session.run(None, {"input": tensor})[0]
        probs = np.exp(out) / np.sum(np.exp(out))
        top_idx = int(np.argmax(probs))
        cls = CLASSES[top_idx]
        conf = float(probs[0][top_idx])

        return cls, conf, CLASS_LABELS.get(cls, cls), CLASS_ICONS.get(cls, "🔊")


def record_from_mic(duration=DURATION):
    """Record audio via pw-record (PipeWire native), then load and resample."""
    import subprocess
    import tempfile
    import librosa

    NATIVE_SR = 48000
    tmp_path = tempfile.mktemp(suffix=".wav")

    print(f"  [Mic] Recording {duration}s via pw-record (target {MIC_TARGET_ID})...")

    proc = subprocess.Popen([
        "pw-record",
        "--target", MIC_TARGET_ID,
        "--rate", str(NATIVE_SR),
        "--channels", "1",
        tmp_path,
    ])

    try:
        proc.wait(timeout=duration)
    except subprocess.TimeoutExpired:
        proc.terminate()   # graceful stop so the WAV header gets finalized
        proc.wait(timeout=3)

    audio, sr = librosa.load(tmp_path, sr=NATIVE_SR, mono=True)
    os.remove(tmp_path)

    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio


def load_wav_file(wav_path):
    """Load a WAV file and resample to TARGET_SR."""
    import librosa

    audio, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    return audio


if __name__ == "__main__":
    print("=" * 55)
    print("  Edge Classifier — Quick Test")
    print("=" * 55)

    clf = EdgeClassifier()

    # Test with mic
    audio = record_from_mic(4.0)
    cls, conf, label, icon = clf.predict(audio)
    print(f"\n  Prediction: {icon} {label} ({conf*100:.1f}%)")

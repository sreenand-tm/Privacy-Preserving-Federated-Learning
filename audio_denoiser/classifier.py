"""
classifier.py
High-Performance Audio Classifier using Review 3's Best ONNX Model
(MobileNetV2 / YAMNet V4 Improved)
"""

import os
import time
import numpy as np
import librosa
import onnxruntime as ort

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(CURRENT_DIR, "models", "model_v4_improved.onnx")
FALLBACK_MODEL_PATH = os.path.join(CURRENT_DIR, "models", "model_v3_robust_yamnet.onnx")

CLASSES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark",
    "drilling", "engine_idling", "gun_shot", "jackhammer",
    "siren", "street_music", "background_noise"
]

CLASS_DISPLAY = {
    "air_conditioner": "Air Conditioner",
    "car_horn": "Car Horn",
    "children_playing": "Children Playing",
    "dog_bark": "Dog Bark",
    "drilling": "Drilling",
    "engine_idling": "Engine Idling",
    "gun_shot": "Gun Shot",
    "jackhammer": "Jackhammer",
    "siren": "Siren",
    "street_music": "Street Music",
    "background_noise": "Background Noise"
}

CLASS_ICONS = {
    "air_conditioner": "[AC]",
    "car_horn": "[HORN]",
    "children_playing": "[CHILD]",
    "dog_bark": "[DOG]",
    "drilling": "[DRILL]",
    "engine_idling": "[ENGINE]",
    "gun_shot": "[SHOT]",
    "jackhammer": "[JACKHAMMER]",
    "siren": "[SIREN]",
    "street_music": "[MUSIC]",
    "background_noise": "[NOISE]"
}

TARGET_SR = 22050
DURATION = 4.0
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

class AudioClassifier:
    def __init__(self, model_path=None):
        if model_path is None:
            if os.path.exists(DEFAULT_MODEL_PATH):
                model_path = DEFAULT_MODEL_PATH
            elif os.path.exists(FALLBACK_MODEL_PATH):
                model_path = FALLBACK_MODEL_PATH
            else:
                raise FileNotFoundError("No trained ONNX model found in models/ directory!")
                
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        
        # Initialize ONNX runtime session
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(self.model_path, opts)
        self.input_name = self.session.get_inputs()[0].name
        
    def extract_mel(self, audio, sr):
        """Extract log Mel-spectrogram normalized to [0, 1]."""
        # Resample to 22050 Hz if needed
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
            
        target_len = int(TARGET_SR * DURATION)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
            
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=TARGET_SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmin=0.0,
            fmax=TARGET_SR / 2.0
        )
        log_mel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
        norm_mel = (log_mel + 80.0) / 80.0
        norm_mel = np.clip(norm_mel, 0.0, 1.0)
        return norm_mel.astype(np.float32)

    def predict(self, audio, sr=TARGET_SR):
        """
        Run inference on audio array.
        Returns:
            dict with:
                - class_name: internal class key
                - display_name: formatted name
                - icon: emoji icon
                - confidence: float (0.0 to 1.0)
                - top3: list of (display_name, confidence)
                - latency_ms: execution time in milliseconds
        """
        t0 = time.time()
        mel = self.extract_mel(audio, sr)
        # Reshape to [1, 1, 128, T]
        tensor = np.expand_dims(np.expand_dims(mel, axis=0), axis=0)
        
        logits = self.session.run(None, {self.input_name: tensor})[0][0]
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        pred_class = CLASSES[best_idx]
        
        # Top 3
        sorted_indices = np.argsort(probs)[::-1][:3]
        top3 = [(CLASS_DISPLAY.get(CLASSES[i], CLASSES[i]), float(probs[i])) for i in sorted_indices]
        
        latency = (time.time() - t0) * 1000.0
        
        return {
            "class_name": pred_class,
            "display_name": CLASS_DISPLAY.get(pred_class, pred_class),
            "icon": CLASS_ICONS.get(pred_class, "🔊"),
            "confidence": confidence,
            "top3": top3,
            "latency_ms": latency
        }

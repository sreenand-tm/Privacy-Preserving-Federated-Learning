"""
Evaluate and compare BOTH the 5-Client and 8-Client FL Robust Models on custom audio files.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np

from config import DEVICE, NUM_CLASSES, CLASS_NAMES, SAMPLE_RATE, AUDIO_DURATION
from model import UrbanSoundCNN
from preprocess import load_and_resample, compute_mel_spectrogram

KEYWORD_TO_CLASS = {
    "siren": "siren",
    "horn": "car_horn",
    "jack": "jackhammer",
    "air": "air_conditioner",
    "engine": "engine_idling",
    "bark": "dog_bark",
    "dog": "dog_bark",
    "drill": "drilling",
    "gun": "gun_shot",
    "children": "children_playing",
    "music": "street_music"
}

def get_expected_class(filename):
    lower = filename.lower()
    for kw, target_class in KEYWORD_TO_CLASS.items():
        if kw in lower:
            return target_class
    return "unknown"

def load_model(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return None
    model = UrbanSoundCNN(num_classes=NUM_CLASSES).to(DEVICE)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict_sample(model, audio_tensor):
    with torch.no_grad():
        output = model(audio_tensor)
        probs = F.softmax(output, dim=1).squeeze().cpu().numpy()
    pred_idx = np.argmax(probs)
    return CLASS_NAMES[pred_idx], probs[pred_idx] * 100

def main():
    path_5client = "fed/global_model_robust.pth"
    path_8client = "8_train/global_model_robust.pth"
    sounds_dir = "sounds"

    model_5 = load_model(path_5client)
    model_8 = load_model(path_8client)

    print("=" * 95)
    print("  FEDERATED LEARNING MODEL COMPARISON TEST (5-CLIENT vs 8-CLIENT)")
    print("=" * 95)
    print(f"  5-Client Model: {path_5client} ({'LOADED' if model_5 else 'NOT FOUND'})")
    print(f"  8-Client Model: {path_8client} ({'LOADED' if model_8 else 'NOT FOUND'})")
    print(f"  Audio Folder:   {sounds_dir}")
    print("-" * 95)

    files = [f for f in os.listdir(sounds_dir) if f.endswith(('.mp3', '.wav', '.flac'))]
    if not files:
        print("No audio files found!")
        return

    c5_correct, c8_correct, total = 0, 0, 0

    header = f"{'Filename':<35} | {'Expected':<13} | {'5-Client Pred (Conf)':<20} | {'8-Client Pred (Conf)':<20}"
    print(header)
    print("-" * 95)

    for file_name in files:
        file_path = os.path.join(sounds_dir, file_name)
        expected = get_expected_class(file_name)

        try:
            y = load_and_resample(file_path, SAMPLE_RATE)
            target_length = int(SAMPLE_RATE * AUDIO_DURATION)

            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), mode='constant')
            elif len(y) > target_length:
                best_window = y[:target_length]
                max_energy = -1.0
                hop = int(SAMPLE_RATE * 0.5)
                for start in range(0, len(y) - target_length + 1, hop):
                    window = y[start : start + target_length]
                    energy = np.mean(window ** 2)
                    if energy > max_energy:
                        max_energy = energy
                        best_window = window
                y = best_window

            log_mel_spec = compute_mel_spectrogram(y, SAMPLE_RATE)
            audio_tensor = torch.tensor(log_mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

            # Predict 5-Client
            if model_5:
                p5_cls, p5_conf = predict_sample(model_5, audio_tensor)
                m5_icon = "✅" if p5_cls == expected else "❌"
                if p5_cls == expected: c5_correct += 1
                res_5 = f"{p5_cls} ({p5_conf:.0f}%) {m5_icon}"
            else:
                res_5 = "N/A"

            # Predict 8-Client
            if model_8:
                p8_cls, p8_conf = predict_sample(model_8, audio_tensor)
                m8_icon = "✅" if p8_cls == expected else "❌"
                if p8_cls == expected: c8_correct += 1
                res_8 = f"{p8_cls} ({p8_conf:.0f}%) {m8_icon}"
            else:
                res_8 = "N/A"

            total += 1
            disp_name = (file_name[:32] + '...') if len(file_name) > 35 else file_name
            print(f"{disp_name:<35} | {expected:<13} | {res_5:<20} | {res_8:<20}")

        except Exception as e:
            print(f"{file_name:<35} | ERROR: {e}")

    print("=" * 95)
    if total > 0:
        acc_5 = (c5_correct / total) * 100 if model_5 else 0
        acc_8 = (c8_correct / total) * 100 if model_8 else 0
        print(f"  ACCURACY SUMMARY:")
        print(f"  👉 5-Client Model: {c5_correct}/{total} Correct ({acc_5:.1f}%)")
        print(f"  👉 8-Client Model: {c8_correct}/{total} Correct ({acc_8:.1f}%)")
    print("=" * 95 + "\n")

if __name__ == "__main__":
    main()

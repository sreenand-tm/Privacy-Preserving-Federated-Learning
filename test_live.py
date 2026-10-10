import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

from config import DEVICE, NUM_CLASSES, CLASS_NAMES, PROJECT_ROOT
from model import UrbanSoundCNN
from preprocess import extract_mel_spectrogram

def test_live(wav_path):
    print(f"Loading {wav_path}...")
    if not os.path.exists(wav_path):
        print(f"Error: File not found at {wav_path}")
        return

    # 1. Extract features using the EXACT same path as training
    try:
        mel_spec = extract_mel_spectrogram(wav_path)
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return

    # Preprocessor returns (N_MELS, N_FRAMES)
    # Model expects (BATCH, CHANNELS, N_MELS, N_FRAMES) -> (1, 1, 128, 173)
    mel_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # 2. Load the best model (using Fold 5 since it got 86.11% accuracy in our run)
    model = UrbanSoundCNN(num_classes=NUM_CLASSES).to(DEVICE)
    model_path = os.path.join(PROJECT_ROOT, "results", "centralized", "fold5", "best_model.pth")
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find model at {model_path}")
        print("Did you delete the 'results' folder?")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()

    # 3. Inference
    with torch.no_grad():
        output = model(mel_tensor)
        probs = F.softmax(output, dim=1).squeeze().cpu().numpy()
        
    # 4. Display results
    print(f"\n{'-'*40}")
    print(f"  RESULTS FOR: {os.path.basename(wav_path)}")
    print(f"{'-'*40}")
    
    predicted_idx = np.argmax(probs)
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = probs[predicted_idx] * 100
    
    print(f"  PREDICTED: {predicted_class.upper()} ({confidence:.1f}% confidence)\n")
    
    print("  Top 3 matches:")
    top3_indices = np.argsort(probs)[::-1][:3]
    for i, idx in enumerate(top3_indices):
        print(f"    {i+1}. {CLASS_NAMES[idx]:<20} {probs[idx]*100:>5.1f}%")
    print(f"{'-'*40}\n")
        
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_live.py <path_to_wav_file>")
        print("Example: python test_live.py live_dog_bark_1.wav")
    else:
        test_live(sys.argv[1])

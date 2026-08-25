"""
Node 1 (Physical PC Node) — Review 1 Demo
Runs Physical Local PyTorch Training & Backpropagation on PC.
Connects to Master PC Dashboard (http://127.0.0.1:5000), fetches global weights, 
trains locally, applies FedProx + Differential Privacy, and uploads weight deltas (delta_w).
"""

import os
import sys
import io
import time
import copy
import urllib.request
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset

PC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(PC_DIR))
sys.path.append(PC_DIR)
sys.path.append(PROJECT_ROOT)

from config import DEVICE, LEARNING_RATE, WEIGHT_DECAY, NUM_CLASSES
from model import UrbanSoundCNN
from dataset import get_fold_dataloaders

SERVER_URL = "http://127.0.0.1:5000"
NODE_ID = 1
LOCAL_EPOCHS = 3
BATCH_SIZE = 32

def partition_data_dirichlet(dataset, num_clients=5, alpha=0.5):
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        np.random.shuffle(idx_c)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions * len(idx_c)).astype(int)
        diff = len(idx_c) - proportions.sum()
        for i in range(diff):
            proportions[i % num_clients] += 1
        start = 0
        for client_id, count in enumerate(proportions):
            client_indices[client_id].extend(idx_c[start:start+count])
            start += count
    return client_indices

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def spec_augment(spec, max_time_mask=20, max_freq_mask=15):
    cloned = spec.clone()
    num_mel_channels = cloned.size(2)
    num_frames = cloned.size(3)
    f = int(np.random.uniform(0, max_freq_mask))
    f0 = int(np.random.uniform(0, num_mel_channels - f))
    cloned[:, :, f0:f0+f, :] = 0
    t = int(np.random.uniform(0, max_time_mask))
    t0 = int(np.random.uniform(0, num_frames - t))
    cloned[:, :, :, t0:t0+t] = 0
    return cloned

def download_global_model():
    req = urllib.request.Request(f"{SERVER_URL}/get_global_model")
    with urllib.request.urlopen(req) as response:
        model_bytes = response.read()
    buffer = io.BytesIO(model_bytes)
    state_dict = torch.load(buffer, map_location='cpu', weights_only=False)
    model = UrbanSoundCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(state_dict)
    return model

def upload_update(update_delta):
    buffer = io.BytesIO()
    torch.save(update_delta, buffer)
    payload = buffer.getvalue()
    
    req = urllib.request.Request(
        f"{SERVER_URL}/upload_update",
        data=payload,
        headers={
            "Content-Type": "application/octet-stream",
            "X-Client-ID": str(NODE_ID)
        },
        method="POST"
    )
    with urllib.request.urlopen(req) as response:
        return response.read().decode('utf-8')

def send_heartbeat():
    try:
        req = urllib.request.Request(
            f"{SERVER_URL}/heartbeat",
            data=b"",
            headers={"X-Client-ID": str(NODE_ID)},
            method="POST"
        )
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        pass

def run_node1_pc():
    send_heartbeat()
    print("=" * 65)
    print(f"  NODE 1 (PHYSICAL PC NODE)")
    print("=" * 65)
    
    train_loader, _ = get_fold_dataloaders(10)
    client_indices = partition_data_dirichlet(train_loader.dataset, 5, 0.5)
    subset = Subset(train_loader.dataset, client_indices[NODE_ID - 1])
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"\n[NODE 1 PC] Fetching global model from server...")
    global_model = download_global_model()
    
    model = copy.deepcopy(global_model).to(DEVICE)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    print(f"[NODE 1 PC] Running PyTorch Backpropagation ({LOCAL_EPOCHS} local epochs)...")
    for epoch in range(LOCAL_EPOCHS):
        running_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            inputs = spec_augment(inputs)
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=0.2)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            
            # FedProx Proximal Term
            prox_term = 0.0
            for w, w_t in zip(model.parameters(), global_model.parameters()):
                prox_term += (w - w_t.to(DEVICE)).pow(2).sum()
            loss += (0.01 / 2) * prox_term
            
            loss.backward()  # AUTOGRAD BACKPROPAGATION
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            
        print(f"  Epoch {epoch+1}/{LOCAL_EPOCHS} | Loss: {running_loss/len(loader):.4f}")
        
    # Compute Update Delta + Differential Privacy
    update = {}
    for (name, new_w), (_, old_w) in zip(model.state_dict().items(), global_model.state_dict().items()):
        delta = new_w.cpu() - old_w.cpu()
        if torch.is_floating_point(delta):
            clip_threshold = 10.0
            norm = torch.norm(delta)
            scale = min(1.0, float(clip_threshold / (norm.item() + 1e-6)))
            delta = delta * scale
            delta += torch.randn_like(delta) * 0.005
        update[name] = delta

    print(f"[NODE 1 PC] Uploading weight update delta (delta_w) to PC Master Dashboard...")
    upload_update(update)
    print("[SUCCESS] [NODE 1 PC] Update uploaded successfully!")

if __name__ == "__main__":
    run_node1_pc()

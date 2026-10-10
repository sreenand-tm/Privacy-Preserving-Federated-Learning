"""
PC Nodes Client Script (Runs Virtual Clients 0, 1, 2 and PC Node 3)
Connects to fl_server.py via HTTP, downloads global model, runs PyTorch backprop + FedProx, and uploads updates.
"""

import os
import sys
import io
import time
import copy
import json
import urllib.request
import urllib.parse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset

# Ensure imports work from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEVICE, LEARNING_RATE, WEIGHT_DECAY, NUM_CLASSES
from model import UrbanSoundCNN
from dataset import get_fold_dataloaders
from train_federated import partition_data_dirichlet
from train_centralized import spec_augment, mixup_data, mixup_criterion

SERVER_URL = "http://127.0.0.1:5000"
LOCAL_EPOCHS = 3
BATCH_SIZE = 32

def download_global_model():
    req = urllib.request.Request(f"{SERVER_URL}/get_global_model")
    with urllib.request.urlopen(req) as response:
        model_bytes = response.read()
    buffer = io.BytesIO(model_bytes)
    state_dict = torch.load(buffer, map_location='cpu', weights_only=False)
    model = UrbanSoundCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(state_dict)
    return model

def upload_update(client_id, update_delta):
    buffer = io.BytesIO()
    torch.save(update_delta, buffer)
    payload = buffer.getvalue()
    
    req = urllib.request.Request(
        f"{SERVER_URL}/upload_update",
        data=payload,
        headers={
            "Content-Type": "application/octet-stream",
            "X-Client-ID": str(client_id)
        },
        method="POST"
    )
    with urllib.request.urlopen(req) as response:
        return response.read().decode('utf-8')

def train_client_local(client_id, dataloader, is_malicious=False):
    print(f"\n[CLIENT {client_id}] Fetching global model from server...")
    global_model = download_global_model()
    
    model = copy.deepcopy(global_model).to(DEVICE)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    print(f"[CLIENT {client_id}] Running PyTorch Backpropagation ({LOCAL_EPOCHS} local epochs)...")
    for epoch in range(LOCAL_EPOCHS):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            inputs = spec_augment(inputs)
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=0.2)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            
            # FedProx term
            prox_term = 0.0
            for w, w_t in zip(model.parameters(), global_model.parameters()):
                prox_term += (w - w_t.to(DEVICE)).pow(2).sum()
            loss += (0.01 / 2) * prox_term
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
    # Calculate update delta
    update = {}
    for (name, new_w), (_, old_w) in zip(model.state_dict().items(), global_model.state_dict().items()):
        delta = new_w.cpu() - old_w.cpu()
        is_floating = torch.is_floating_point(delta)
        
        if is_floating:
            # DP Clipping + Noise
            clip_threshold = 10.0
            norm = torch.norm(delta)
            scale = min(1.0, float(clip_threshold / (norm.item() + 1e-6)))
            delta = delta * scale
            delta += torch.randn_like(delta) * 0.005
            
        if is_malicious and is_floating:
            delta = -5.0 * delta # Byzantine attack
            
        update[name] = delta

    print(f"[CLIENT {client_id}] Uploading update delta (Δw) to server...")
    upload_update(client_id, update)
    print(f"✅ [CLIENT {client_id}] Round completed successfully.")

def run_pc_nodes():
    train_loader, _ = get_fold_dataloaders(10)
    client_indices = partition_data_dirichlet(train_loader.dataset, 5, 0.5)
    
    print("=" * 65)
    print("  PC CLIENT NODES (Clients 0, 1, 2, 3)")
    print("=" * 65)
    
    # We will train PC Clients (0, 1, 2, 3) for the current round
    for client_id in range(4):
        subset = Subset(train_loader.dataset, client_indices[client_id])
        loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
        train_client_local(client_id, loader, is_malicious=(client_id == 2))

if __name__ == "__main__":
    run_pc_nodes()

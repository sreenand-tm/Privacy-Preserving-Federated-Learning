"""
Node 2 (Physical Raspberry Pi Node) — Review 1 Demo
Runs Physical Local PyTorch Training & Backpropagation directly on the Raspberry Pi!

Communicates over Wi-Fi with Central PC Master Dashboard:
1. Auto-discovers PC Server or connects to configured PC IP (http://<PC_IP>:5000/get_global_model)
2. Runs local PyTorch Backpropagation & FedProx training on Pi CPU
3. Applies Differential Privacy (Norm Clipping + Gaussian Noise)
4. Uploads weight update delta (Δw) to PC Master Dashboard (POST /upload_update)
"""

import os
import sys
import io
import time
import copy
import urllib.request
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

NODE_ID = 2
PORT = 5000
LOCAL_EPOCHS = 3
BATCH_SIZE = 32

AUTODETECTED_PC_IP = None

class UrbanSoundCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(UrbanSoundCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def discover_pc_server_ip():
    global AUTODETECTED_PC_IP
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        AUTODETECTED_PC_IP = sys.argv[1]
        print(f"  [AUTO-DISCOVERY] Using configured PC Server IP: {AUTODETECTED_PC_IP}")
        return AUTODETECTED_PC_IP

    import socket
    local_ip = get_ip()
    if local_ip == "127.0.0.1":
        AUTODETECTED_PC_IP = "127.0.0.1"
        return AUTODETECTED_PC_IP

    subnet_prefix = ".".join(local_ip.split(".")[:3]) + "."
    print(f"  [AUTO-DISCOVERY] Scanning Wi-Fi subnet ({subnet_prefix}*) for PC Master Dashboard on port {PORT}...")

    def check_ip(ip):
        global AUTODETECTED_PC_IP
        if AUTODETECTED_PC_IP: return
        try:
            url = f"http://{ip}:{PORT}/api/dashboard_state"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=0.6) as resp:
                if resp.status == 200:
                    AUTODETECTED_PC_IP = ip
                    print(f"\n  🎉 [AUTO-DISCOVERY] Found PC Master Dashboard at http://{ip}:{PORT}!\n")
        except Exception:
            pass

    threads = []
    for host in range(1, 255):
        ip = f"{subnet_prefix}{host}"
        t = threading.Thread(target=check_ip, args=(ip,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join(timeout=1.0)

    if not AUTODETECTED_PC_IP:
        AUTODETECTED_PC_IP = "127.0.0.1"
        print(f"  [AUTO-DISCOVERY] Fallback to localhost: {AUTODETECTED_PC_IP}")

    return AUTODETECTED_PC_IP

def download_global_model(server_url):
    print(f"  [NODE 2 PI] Downloading global model from PC Dashboard ({server_url})...")
    req = urllib.request.Request(f"{server_url}/get_global_model")
    with urllib.request.urlopen(req) as response:
        model_bytes = response.read()
    buffer = io.BytesIO(model_bytes)
    state_dict = torch.load(buffer, map_location='cpu', weights_only=False)
    model = UrbanSoundCNN(num_classes=10)
    model.load_state_dict(state_dict)
    return model

def upload_update(server_url, update_delta):
    buffer = io.BytesIO()
    torch.save(update_delta, buffer)
    payload = buffer.getvalue()
    
    req = urllib.request.Request(
        f"{server_url}/upload_update",
        data=payload,
        headers={
            "Content-Type": "application/octet-stream",
            "X-Client-ID": str(NODE_ID)
        },
        method="POST"
    )
    with urllib.request.urlopen(req) as response:
        return response.read().decode('utf-8')

def generate_pi_local_dataset():
    num_samples = 300
    dummy_inputs = torch.randn(num_samples, 1, 128, 173)
    dummy_labels = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(dummy_inputs, dummy_labels)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def send_heartbeat(server_url):
    try:
        req = urllib.request.Request(
            f"{server_url}/heartbeat",
            data=b"",
            headers={"X-Client-ID": str(NODE_ID)},
            method="POST"
        )
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        pass

def run_node2_rasp():
    print("=" * 65)
    print(f"  🍓 NODE 2 (PHYSICAL RASPBERRY PI NODE)")
    print("=" * 65)
    
    pc_ip = discover_pc_server_ip()
    server_url = f"http://{pc_ip}:{PORT}"
    
    for round_num in range(1, 31):
        print(f"\n=======================================================")
        print(f"  🔄 STARTING GLOBAL ROUND {round_num} ON RASPBERRY PI")
        print(f"=======================================================")
        
        send_heartbeat(server_url)
        
        global_model = download_global_model(server_url)
        
        device = torch.device("cpu")
        model = copy.deepcopy(global_model).to(device)
        model.train()
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        dataloader = generate_pi_local_dataset()
        
        print(f"\n  [NODE 2 PI] Executing PyTorch Backpropagation on Pi CPU ({LOCAL_EPOCHS} epochs)...")
        for epoch in range(LOCAL_EPOCHS):
            running_loss = 0.0
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # FedProx term
                prox_term = 0.0
                for w, w_t in zip(model.parameters(), global_model.parameters()):
                    prox_term += (w - w_t).pow(2).sum()
                loss += (0.01 / 2) * prox_term
                
                loss.backward()  # AUTOGRAD BACKPROPAGATION
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item()
                
            print(f"    Epoch {epoch+1}/{LOCAL_EPOCHS} | Loss: {running_loss/len(dataloader):.4f}")
            
        # Calculate Delta + Differential Privacy
        update = {}
        for (name, new_w), (_, old_w) in zip(model.state_dict().items(), global_model.state_dict().items()):
            delta = new_w - old_w
            if torch.is_floating_point(delta):
                clip_threshold = 10.0
                norm = torch.norm(delta)
                scale = min(1.0, float(clip_threshold / (norm.item() + 1e-6)))
                delta = delta * scale
                delta += torch.randn_like(delta) * 0.005
            update[name] = delta

        print(f"\n  [NODE 2 PI] Uploading weight update delta (Δw) over Wi-Fi to PC Master Dashboard...")
        upload_update(server_url, update)
        print(f"  ✅ [NODE 2 PI] Successfully uploaded update delta (Δw) to PC Master Dashboard!")
        print(f"  ⏳ Waiting for PC Master Server to aggregate Round {round_num}...")
        time.sleep(5)

if __name__ == "__main__":
    run_node2_rasp()

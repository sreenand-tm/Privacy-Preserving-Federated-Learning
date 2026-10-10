"""
Central Networked Federated Learning Server (Runs on PC)
Uses standard library http.server (NO external Flask required!)

Coordinates 5 Real/Virtual Nodes over Network:
- Node 0, 1, 2: Virtual Clients on PC
- Node 3: Physical PC Client
- Node 4: Physical Raspberry Pi Client

Endpoints:
- GET /get_global_model : Serves current global model weights (.pth)
- GET /status           : Shows round status & connected clients
- POST /upload_update   : Receives client update delta Δw via HTTP POST
"""

import http.server
import socketserver
import os
import sys
import io
import json
import socket
import torch
import numpy as np

# Ensure imports work from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEVICE, NUM_CLASSES, PROJECT_ROOT
from model import UrbanSoundCNN
from dataset import get_fold_dataloaders

PORT = 5000
NUM_CLIENTS = 5
EXPECTED_UPDATES_PER_ROUND = 5
NUM_ROUNDS = 30

# Global State
current_round = 1
received_updates = {}
global_model = None
test_loader = None
acc_history = []

def aggregate_krum(updates, f=1):
    """Multi-Krum: Drops 'f' outliers, averages the rest based on Euclidean distance."""
    flat_updates = []
    for u in updates:
        flat_updates.append(torch.cat([v.float().flatten() for v in u.values() if torch.is_floating_point(v)]))
        
    n = len(updates)
    scores = np.zeros(n)
    
    for i in range(n):
        dists = []
        for j in range(n):
            if i == j: continue
            dist = torch.norm(flat_updates[i] - flat_updates[j]).item()
            dists.append(dist)
        dists.sort()
        scores[i] = sum(dists[:max(1, n - f - 2)])
        
    top_k = n - f
    best_indices = np.argsort(scores)[:top_k]
    dropped_indices = np.argsort(scores)[top_k:]
    
    print(f"\n  [KRUM AGGREGATION] Dropped Client(s): {list(dropped_indices)} | Kept: {list(best_indices)}")
    
    avg_update = {}
    for key in updates[0].keys():
        avg_update[key] = sum([updates[idx][key] for idx in best_indices]) / len(best_indices)
    return avg_update

def apply_update(model, update):
    state_dict = model.state_dict()
    for key in state_dict.keys():
        state_dict[key] += update[key].to(DEVICE, dtype=state_dict[key].dtype)
    model.load_state_dict(state_dict)

def evaluate_global_model(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total

def process_round_aggregation():
    global current_round, received_updates, global_model, test_loader, acc_history
    
    print(f"\n" + "=" * 65)
    print(f"  PROCESSING ROUND {current_round}/{NUM_ROUNDS} AGGREGATION ({len(received_updates)} UPDATES)")
    print("=" * 65)
    
    # Sort updates by client id
    client_ids = sorted(received_updates.keys())
    updates_list = [received_updates[cid] for cid in client_ids]
    
    # Multi-Krum Aggregation
    aggregated_update = aggregate_krum(updates_list, f=1)
    apply_update(global_model, aggregated_update)
    
    # Evaluate Accuracy
    acc = evaluate_global_model(global_model, test_loader)
    acc_history.append(acc)
    print(f"  🏆 Round {current_round} Global Model Test Accuracy: {acc:.2f}%\n")
    
    # Save Model
    save_path = os.path.join(PROJECT_ROOT, "network_fl", "global_model_robust.pth")
    torch.save(global_model.state_dict(), save_path)
    print(f"  Saved global model to {save_path}")
    
    # Reset for next round
    received_updates = {}
    current_round += 1

class FLServerHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return # Suppress default http log clutter

    def do_GET(self):
        global current_round, received_updates, global_model
        
        if self.path == "/get_global_model":
            buffer = io.BytesIO()
            torch.save(global_model.state_dict(), buffer)
            model_bytes = buffer.getvalue()
            
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(model_bytes)))
            self.end_headers()
            self.wfile.write(model_bytes)
            
        elif self.path == "/status":
            status_data = {
                "round": current_round,
                "received_updates": list(received_updates.keys()),
                "total_expected": EXPECTED_UPDATES_PER_ROUND
            }
            body = json.dumps(status_data).encode('utf-8')
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        global current_round, received_updates
        
        if self.path == "/upload_update":
            content_length = int(self.headers['Content-Length'])
            client_id = int(self.headers.get('X-Client-ID', 0))
            
            post_data = self.rfile.read(content_length)
            buffer = io.BytesIO(post_data)
            update_delta = torch.load(buffer, map_location='cpu', weights_only=False)
            
            received_updates[client_id] = update_delta
            print(f"  📥 Received Update from Client {client_id} (Round {current_round}). Total: {len(received_updates)}/{EXPECTED_UPDATES_PER_ROUND}")
            
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Update received successfully")
            
            # Trigger aggregation if all expected updates arrive
            if len(received_updates) >= EXPECTED_UPDATES_PER_ROUND:
                process_round_aggregation()
        else:
            self.send_response(404)
            self.end_headers()

def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def start_server():
    global global_model, test_loader
    print("=" * 65)
    print("  CENTRALIZED NETWORKED FEDERATED LEARNING SERVER")
    print("=" * 65)
    
    local_ip = get_ip()
    print(f"  Server IP: http://{local_ip}:{PORT}")
    print("  Listening for 5 Clients (PC Virtuals, PC Node & Raspberry Pi Node)...\n")
    
    _, test_loader = get_fold_dataloaders(10)
    global_model = UrbanSoundCNN(num_classes=NUM_CLASSES).to(DEVICE)
    
    # Save initial model
    save_path = os.path.join(PROJECT_ROOT, "network_fl", "global_model_robust.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(global_model.state_dict(), save_path)
    
    with socketserver.TCPServer(("", PORT), FLServerHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    start_server()

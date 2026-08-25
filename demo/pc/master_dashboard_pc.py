"""
Central PC Master Dashboard & FL Aggregator Server (Review 1 Demo)
Coordinates all 5 Nodes:
- Node 1: Physical PC Node
- Node 2: Physical Raspberry Pi Node (over Wi-Fi)
- Node 3, 4, 5: Virtual Baseline Dataset Nodes on PC

Serves live visual web dashboard at http://localhost:5000
"""

import http.server
import socketserver
import os
import sys
import io
import json
import socket
import time
import threading
import torch
import numpy as np

# Ensure imports work locally from inside the pc/ directory
PC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(PC_DIR))
sys.path.append(PC_DIR)
sys.path.append(PROJECT_ROOT)

from config import DEVICE, NUM_CLASSES
from model import UrbanSoundCNN
from dataset import get_fold_dataloaders

PORT = 5000
NUM_CLIENTS = 5
EXPECTED_UPDATES = 5
NUM_ROUNDS = 30

# Shared Live Dashboard State
current_round = 1
received_updates = {}
node_statuses = {
    1: {"name": "Node 1 (Physical PC)", "status": "Disconnected", "last_seen": "Never", "type": "Physical PC"},
    2: {"name": "Node 2 (Physical Raspberry Pi)", "status": "Disconnected", "last_seen": "Never", "type": "Physical Pi"},
    3: {"name": "Node 3 (Virtual Baseline 1)", "status": "Disconnected", "last_seen": "Never", "type": "Virtual PC"},
    4: {"name": "Node 4 (Virtual Baseline 2)", "status": "Disconnected", "last_seen": "Never", "type": "Virtual PC"},
    5: {"name": "Node 5 (Virtual Baseline 3)", "status": "Disconnected", "last_seen": "Never", "type": "Virtual PC"}
}
update_logs = []
krum_history = []
acc_history = []
global_model = None
test_loader = None

def aggregate_krum(updates, f=1):
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
    return list(best_indices), list(dropped_indices)

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
    return 100.0 * correct / total if total > 0 else 0.0

def process_round_aggregation():
    global current_round, received_updates, global_model, test_loader, krum_history, acc_history
    
    print(f"\n" + "=" * 65)
    print(f"  PROCESSING ROUND {current_round}/{NUM_ROUNDS} AGGREGATION")
    print("=" * 65)
    
    client_ids = sorted(received_updates.keys())
    updates_list = [received_updates[cid] for cid in client_ids]
    
    best_indices, dropped_indices = aggregate_krum(updates_list, f=1)
    
    kept_clients = [client_ids[i] for i in best_indices]
    dropped_clients = [client_ids[i] for i in dropped_indices]
    
    krum_record = {
        "round": current_round,
        "kept_nodes": kept_clients,
        "dropped_nodes": dropped_clients,
        "timestamp": time.strftime("%H:%M:%S")
    }
    krum_history.append(krum_record)
    
    # Apply average update of kept nodes
    avg_update = {}
    for key in updates_list[0].keys():
        avg_update[key] = sum([updates_list[idx][key] for idx in best_indices]) / len(best_indices)
        
    state_dict = global_model.state_dict()
    for key in state_dict.keys():
        state_dict[key] += avg_update[key].to(DEVICE, dtype=state_dict[key].dtype)
    global_model.load_state_dict(state_dict)
    
    # Evaluate
    acc = evaluate_global_model(global_model, test_loader)
    acc_history.append({"round": current_round, "accuracy": round(acc, 2)})
    print(f"  [SUCCESS] Round {current_round} Accuracy: {acc:.2f}% | Kept Nodes: {kept_clients} | Dropped: {dropped_clients}")
    
    # Save checkpoint
    torch.save(global_model.state_dict(), os.path.join(PC_DIR, "global_model_robust.pth"))
    
    received_updates = {}
    current_round += 1

class MasterDashboardHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def end_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_cors_headers()

    def do_GET(self):
        global current_round, received_updates, node_statuses, update_logs, krum_history, acc_history, global_model
        
        if self.path == "/" or self.path == "/index.html":
            html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
            if os.path.exists(html_path):
                with open(html_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(content)))
                self.end_cors_headers()
                self.wfile.write(content)
            else:
                self.send_response(404)
                self.end_headers()

        elif self.path == "/get_global_model":
            buffer = io.BytesIO()
            torch.save(global_model.state_dict(), buffer)
            model_bytes = buffer.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(model_bytes)))
            self.end_cors_headers()
            self.wfile.write(model_bytes)

        elif self.path == "/api/dashboard_state":
            state = {
                "current_round": current_round,
                "num_rounds": NUM_ROUNDS,
                "received_count": len(received_updates),
                "expected_count": EXPECTED_UPDATES,
                "nodes": node_statuses,
                "logs": update_logs[-15:],
                "krum": krum_history[-10:],
                "accuracy": acc_history
            }
            body = json.dumps(state).encode('utf-8')
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_cors_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        global current_round, received_updates, node_statuses, update_logs, global_model
        
        if self.path == "/heartbeat":
            node_id = int(self.headers.get('X-Client-ID', 1))
            node_statuses[node_id]["status"] = "Active"
            node_statuses[node_id]["last_seen"] = time.strftime("%H:%M:%S")
            print(f"  [HEARTBEAT] Node {node_id} registered active!")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_cors_headers()
            self.wfile.write(b"Heartbeat OK")

        elif self.path == "/select_model":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(body)
                model_name = data.get("model_file", "model_5clients.onnx")
                model_path = os.path.join(PC_DIR, model_name)
                pth_equivalent = os.path.join(PC_DIR, model_name.replace('.onnx', '.pth'))
                
                if os.path.exists(model_path):
                    if os.path.exists(pth_equivalent):
                        state_dict = torch.load(pth_equivalent, map_location=DEVICE, weights_only=True)
                        global_model.load_state_dict(state_dict)
                    print(f"\n  [MODEL SWITCH] Active global ONNX model updated to: {model_name}")
                    resp = json.dumps({"success": True, "message": f"Switched active model to {model_name}"}).encode('utf-8')
                else:
                    resp = json.dumps({"success": False, "message": f"Model file {model_name} not found"}).encode('utf-8')
            except Exception as e:
                resp = json.dumps({"success": False, "message": str(e)}).encode('utf-8')

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_cors_headers()
            self.wfile.write(resp)

        elif self.path == "/upload_update":
            content_length = int(self.headers.get('Content-Length', 0))
            node_id = int(self.headers.get('X-Client-ID', 1))
            
            post_data = self.rfile.read(content_length)
            buffer = io.BytesIO(post_data)
            update_delta = torch.load(buffer, map_location='cpu', weights_only=False)
            
            # Compute norm of update vector
            norms = [torch.norm(v.float()).item() for v in update_delta.values() if torch.is_floating_point(v)]
            avg_norm = float(np.mean(norms)) if norms else 0.0
            
            received_updates[node_id] = update_delta
            
            # Update Node status
            node_statuses[node_id]["status"] = "Active"
            node_statuses[node_id]["last_seen"] = time.strftime("%H:%M:%S")
            
            log_entry = {
                "timestamp": time.strftime("%H:%M:%S"),
                "node_id": node_id,
                "node_name": node_statuses[node_id]["name"],
                "round": current_round,
                "delta_norm": round(avg_norm, 4)
            }
            update_logs.append(log_entry)
            
            print(f"  [UPDATE RECEIVE] Node {node_id} (Norm: {avg_norm:.4f}). Total: {len(received_updates)} active nodes.")
            
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_cors_headers()
            self.wfile.write(b"Update received")
            
            # Immediate aggregation if all 5 connected, or trigger background timer for 4 nodes (when Pi is offline)
            if len(received_updates) >= EXPECTED_UPDATES:
                process_round_aggregation()
            elif len(received_updates) >= 4:
                def delayed_aggregation(target_round):
                    time.sleep(3.0)
                    if current_round == target_round and len(received_updates) >= 4:
                        print("\n  [INFO] Node 2 (Raspberry Pi) is offline. Aggregating active PC nodes...")
                        process_round_aggregation()
                threading.Thread(target=delayed_aggregation, args=(current_round,), daemon=True).start()
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

def start_master_dashboard():
    global global_model, test_loader
    print("=" * 70)
    print("  CENTRAL PC MASTER DASHBOARD SERVER (REVIEW 1 DEMO)")
    print("=" * 70)
    
    local_ip = get_ip()
    print(f"\n  [INFO] Live PC Master Dashboard UI:   http://localhost:{PORT}")
    print(f"  [INFO] Wi-Fi Node API Endpoint:      http://{local_ip}:{PORT}\n")
    
    _, test_loader = get_fold_dataloaders(10)
    global_model = UrbanSoundCNN(num_classes=NUM_CLASSES).to(DEVICE)
    
    init_path = os.path.join(PC_DIR, "model_5clients.pth")
    if not os.path.exists(init_path):
        init_path = os.path.join(PC_DIR, "global_model_robust.pth")
    if os.path.exists(init_path):
        state_dict = torch.load(init_path, map_location=DEVICE, weights_only=True)
        global_model.load_state_dict(state_dict)
        print(f"  [INFO] Loaded pre-trained global model weights from: {os.path.basename(init_path)}")
    
    save_path = os.path.join(PC_DIR, "global_model_robust.pth")
    torch.save(global_model.state_dict(), save_path)
    
    with socketserver.TCPServer(("", PORT), MasterDashboardHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    start_master_dashboard()

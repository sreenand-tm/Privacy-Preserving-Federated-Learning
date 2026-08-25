"""
Phase B: Federated Learning Simulation (8 Clients Benchmark).

Simulates 8 clients collaboratively training a global model using a Non-IID 
(Dirichlet alpha=0.5) split of the UrbanSound8K dataset. 

Compares:
1. Standard FedAvg (Clean)
2. Standard FedAvg (With 1 Malicious Attacker: Client 7)
3. Proposed Method: FedProx + Krum Aggregation + Differential Privacy (With 1 Malicious Attacker: Client 7)
"""
import os
import sys
# Ensure imports work from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from config import (
    DEVICE, LEARNING_RATE, WEIGHT_DECAY,
    NUM_CLASSES, PROJECT_ROOT
)
from model import UrbanSoundCNN
from dataset import get_fold_dataloaders
from train_centralized import spec_augment, mixup_data, mixup_criterion

# ─── FL Hyperparameters (8 Clients) ─────────────────────────────────────────
NUM_CLIENTS = 8
ATTACKER_ID = 7
DIRICHLET_ALPHA = 0.5
NUM_ROUNDS = 30
LOCAL_EPOCHS = 3
BATCH_SIZE = 32

CENTRALIZED_CEILING = 80.05  # Fold 10 ceiling

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(RESULTS_DIR, exist_ok=True)
CHECKPOINT_FILE = os.path.join(RESULTS_DIR, "fl_checkpoint_8.pth")

# ─── Data Partitioning (Non-IID) ──────────────────────────────────────────────
def partition_data_dirichlet(train_dataset, num_clients, alpha=0.5):
    labels = train_dataset.metadata['classID'].values
    indices = np.arange(len(train_dataset))
    client_indices = {i: [] for i in range(num_clients)}
    
    for k in range(NUM_CLASSES):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        splits = np.split(idx_k, proportions)
        for i in range(num_clients):
            client_indices[i].extend(splits[i].tolist())
            
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
        
    return client_indices

# ─── Client Definition ────────────────────────────────────────────────────────
class FederatedClient:
    def __init__(self, client_id, dataloader, device):
        self.id = client_id
        self.dataloader = dataloader
        self.device = device
        
    def train(self, global_model, use_prox=False, mu=0.01, use_dp=False, dp_epsilon=1.0, is_malicious=False):
        model = copy.deepcopy(global_model).to(self.device)
        model.train()
        
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        for epoch in range(LOCAL_EPOCHS):
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                inputs = spec_augment(inputs)
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=0.2)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                
                if use_prox:
                    prox_term = 0.0
                    for w, w_t in zip(model.parameters(), global_model.parameters()):
                        prox_term += (w - w_t.to(self.device)).pow(2).sum()
                    loss += (mu / 2) * prox_term
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
        update = {}
        for (name, new_w), (_, old_w) in zip(model.state_dict().items(), global_model.state_dict().items()):
            delta = new_w.cpu() - old_w.cpu()
            is_floating = torch.is_floating_point(delta)
            
            if use_dp and is_floating:
                clip_threshold = 10.0
                norm = torch.norm(delta)
                scale = min(1.0, float(clip_threshold / (norm.item() + 1e-6)))
                delta = delta * scale
                noise = torch.randn_like(delta) * 0.005
                delta += noise
                
            if is_malicious and is_floating:
                delta = -5.0 * delta # Byzantine attack
                
            update[name] = delta
        return update

# ─── Server Aggregation Rules ─────────────────────────────────────────────────
def aggregate_fedavg(updates):
    avg_update = {}
    n = len(updates)
    for key in updates[0].keys():
        avg_update[key] = sum([u[key] for u in updates]) / n
    return avg_update

def aggregate_krum(updates, f=1, verbose=False):
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
    
    if verbose:
        for i in range(n):
            tag = " [MALICIOUS]" if i == ATTACKER_ID else ""
            norm = torch.norm(flat_updates[i]).item()
            print(f"    Client {i}: score={scores[i]:.2f}, update_norm={norm:.4f}{tag}")
        print(f"    Krum DROPPED: Client(s) {list(dropped_indices)}, KEPT: {list(best_indices)}")
    
    avg_update = {}
    for key in updates[0].keys():
        avg_update[key] = sum([updates[idx][key] for idx in best_indices]) / len(best_indices)
    return avg_update

def apply_update(global_model, update):
    state_dict = global_model.state_dict()
    for key in state_dict.keys():
        state_dict[key] += update[key].to(DEVICE, dtype=state_dict[key].dtype)
    global_model.load_state_dict(state_dict)

# ─── Evaluation ───────────────────────────────────────────────────────────────
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total

# ─── Saving and Plotting ──────────────────────────────────────────────────────
def save_checkpoint(phase, round_num, accs_clean, accs_atk, accs_robust, 
                    model_clean, model_atk, model_robust, client_indices):
    checkpoint = {
        'phase': phase,
        'round_num': round_num,
        'accs_clean': accs_clean,
        'accs_atk': accs_atk,
        'accs_robust': accs_robust,
        'client_indices': client_indices,
        'model_clean_state': model_clean.state_dict() if model_clean else None,
        'model_atk_state': model_atk.state_dict() if model_atk else None,
        'model_robust_state': model_robust.state_dict() if model_robust else None
    }
    torch.save(checkpoint, CHECKPOINT_FILE)

def plot_results(fedavg_clean, fedavg_atk, robust_atk):
    plt.figure(figsize=(10, 6))
    rounds = range(1, NUM_ROUNDS + 1)
    
    if fedavg_clean:
        plt.plot(rounds, fedavg_clean, label="FedAvg (Clean)", color='blue', linestyle=':', linewidth=2)
    if fedavg_atk:
        plt.plot(rounds, fedavg_atk, label="FedAvg (With Attacker)", color='orange', linewidth=2)
    if robust_atk:
        plt.plot(rounds, robust_atk, label="FedProx + Krum + DP (With Attacker)", color='green', linewidth=2)
        
    plt.axhline(y=CENTRALIZED_CEILING, color='red', linestyle='--', label=f"Centralized Ceiling ({CENTRALIZED_CEILING}%)")
    
    plt.title(f"8-Client Federated Learning Accuracy (Non-IID, $\\alpha$={DIRICHLET_ALPHA})", fontsize=14)
    plt.xlabel("Communication Round", fontsize=12)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "federated_comparison_8.png"), dpi=150)
    plt.close()
    print(f"\n[PLOT SAVED] Chart available at {os.path.join(RESULTS_DIR, 'federated_comparison_8.png')}")

def save_simulation_results(client_indices, accs_clean, accs_atk, accs_robust, 
                            model_fedavg_clean, model_fedavg_atk, model_robust):
    print("\nSaving final models and logs to disk...")
    if model_fedavg_clean and model_fedavg_clean.state_dict():
        torch.save(model_fedavg_clean.state_dict(), os.path.join(RESULTS_DIR, "global_model_fedavg_clean.pth"))
    if model_fedavg_atk and model_fedavg_atk.state_dict():
        torch.save(model_fedavg_atk.state_dict(), os.path.join(RESULTS_DIR, "global_model_fedavg_atk.pth"))
    if model_robust and model_robust.state_dict():
        torch.save(model_robust.state_dict(), os.path.join(RESULTS_DIR, "global_model_robust.pth"))
    
    safe_indices = {int(k): [int(i) for i in v] for k, v in client_indices.items()}
    with open(os.path.join(RESULTS_DIR, "client_data_partition.json"), "w") as f:
        json.dump(safe_indices, f, indent=4)
        
    with open(os.path.join(RESULTS_DIR, "federated_accuracy_logs.csv"), "w") as f:
        f.write("Round,FedAvg_Clean,FedAvg_Attacker,FedProx_Krum_DP_Attacker\n")
        for r in range(NUM_ROUNDS):
            c_val = f"{accs_clean[r]:.2f}" if r < len(accs_clean) else "N/A"
            a_val = f"{accs_atk[r]:.2f}" if r < len(accs_atk) else "N/A"
            r_val = f"{accs_robust[r]:.2f}" if r < len(accs_robust) else "N/A"
            f.write(f"{r+1},{c_val},{a_val},{r_val}\n")
            
    print(f"All 8-client results successfully saved in: {RESULTS_DIR}")

# ─── Main Simulation ──────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  FEDERATED LEARNING SIMULATOR — 8 CLIENTS (PHASE B)")
    print("=" * 60)
    train_loader, test_loader = get_fold_dataloaders(10)
    
    # State variables (Skip Run 1 & 2, start directly at Run 3)
    phase = 3
    start_round = 1
    accs_clean = []
    accs_atk = []
    accs_robust = []
    
    model_fedavg_clean = UrbanSoundCNN(num_classes=NUM_CLASSES).to(DEVICE)
    model_fedavg_atk = UrbanSoundCNN(num_classes=NUM_CLASSES).to(DEVICE)
    model_robust = UrbanSoundCNN(num_classes=NUM_CLASSES).to(DEVICE)
    
    # Check for resume
    if os.path.exists(CHECKPOINT_FILE):
        print("\n[RESUMING] Found existing checkpoint. Loading state...")
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE, weights_only=False)
        
        phase = checkpoint['phase']
        start_round = checkpoint['round_num'] + 1
        accs_clean = checkpoint['accs_clean']
        accs_atk = checkpoint['accs_atk']
        accs_robust = checkpoint['accs_robust']
        client_indices = checkpoint['client_indices']
        
        if checkpoint['model_clean_state']:
            model_fedavg_clean.load_state_dict(checkpoint['model_clean_state'])
        if checkpoint['model_atk_state']:
            model_fedavg_atk.load_state_dict(checkpoint['model_atk_state'])
        if checkpoint['model_robust_state']:
            model_robust.load_state_dict(checkpoint['model_robust_state'])
            
        print(f"Resuming from Run {phase}, Round {start_round}...")
        
        if start_round > NUM_ROUNDS:
            phase += 1
            start_round = 1
    else:
        print(f"\n[NEW RUN] Creating Non-IID data partitions across {NUM_CLIENTS} clients...")
        client_indices = partition_data_dirichlet(train_loader.dataset, NUM_CLIENTS, DIRICHLET_ALPHA)

    # Build clients
    clients = []
    for i in range(NUM_CLIENTS):
        subset = Subset(train_loader.dataset, client_indices[i])
        loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
        clients.append(FederatedClient(client_id=i, dataloader=loader, device=DEVICE))

    # ---------------- RUN 1 ----------------
    if phase == 1:
        print("\n" + "-" * 50 + "\n  RUN 1: FedAvg (Clean) [8 Clients]\n" + "-" * 50)
        for r in range(start_round, NUM_ROUNDS + 1):
            updates = [c.train(model_fedavg_clean) for c in clients]
            apply_update(model_fedavg_clean, aggregate_fedavg(updates))
            acc = evaluate(model_fedavg_clean, test_loader, DEVICE)
            accs_clean.append(acc)
            if r % 5 == 0 or r == 1: print(f"  Round {r:2d}/{NUM_ROUNDS} | Accuracy: {acc:.2f}%")
            
            save_checkpoint(1, r, accs_clean, accs_atk, accs_robust, 
                            model_fedavg_clean, model_fedavg_atk, model_robust, client_indices)
            
        phase = 2
        start_round = 1

    # ---------------- RUN 2 ----------------
    if phase == 2:
        print("\n" + "-" * 50 + "\n  RUN 2: FedAvg (With Attacker - Client 7) [8 Clients]\n" + "-" * 50)
        for r in range(start_round, NUM_ROUNDS + 1):
            updates = [c.train(model_fedavg_atk, is_malicious=(c.id == ATTACKER_ID)) for c in clients]
            apply_update(model_fedavg_atk, aggregate_fedavg(updates))
            acc = evaluate(model_fedavg_atk, test_loader, DEVICE)
            accs_atk.append(acc)
            if r % 5 == 0 or r == 1: print(f"  Round {r:2d}/{NUM_ROUNDS} | Accuracy: {acc:.2f}%")
            
            save_checkpoint(2, r, accs_clean, accs_atk, accs_robust, 
                            model_fedavg_clean, model_fedavg_atk, model_robust, client_indices)
            
        phase = 3
        start_round = 1

    # ---------------- RUN 3 ----------------
    if phase == 3:
        print("\n" + "-" * 50 + "\n  RUN 3: FedProx + Krum + DP (With Attacker - Client 7) [8 Clients]\n" + "-" * 50)
        for r in range(start_round, NUM_ROUNDS + 1):
            updates = [c.train(model_robust, use_prox=True, mu=0.01, use_dp=True, dp_epsilon=1.0, is_malicious=(c.id == ATTACKER_ID)) for c in clients]
            verbose = True  # Show Krum diagnostics every round
            apply_update(model_robust, aggregate_krum(updates, f=1, verbose=verbose))
            acc = evaluate(model_robust, test_loader, DEVICE)
            accs_robust.append(acc)
            if r % 5 == 0 or r == 1: print(f"  Round {r:2d}/{NUM_ROUNDS} | Accuracy: {acc:.2f}%")
            
            save_checkpoint(3, r, accs_clean, accs_atk, accs_robust, 
                            model_fedavg_clean, model_fedavg_atk, model_robust, client_indices)

    # Final Output and Saving
    if len(accs_robust) == NUM_ROUNDS:
        plot_results(accs_clean, accs_atk, accs_robust)
        save_simulation_results(client_indices, accs_clean, accs_atk, accs_robust,
                                model_fedavg_clean, model_fedavg_atk, model_robust)
        
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)

if __name__ == "__main__":
    main()

"""
Centralized baseline training with 10-fold cross-validation.
Phase A — Optimizations applied (SpecAugment, Mixup, OneCycleLR, etc.)
*FIXED: 3-epoch moving average for early stopping + augmentation annealing.
*FIXED: Removed invalid ensemble evaluation.

Run:  python train_centralized.py
"""
import os
import json
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    NUM_FOLDS, NUM_CLASSES, CLASS_NAMES, PROJECT_ROOT, METADATA_CSV
)
from model import UrbanSoundCNN, count_parameters
from dataset import get_fold_dataloaders, UrbanSoundDataset
from torch.utils.data import DataLoader
import pandas as pd


RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "centralized")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "checkpoint.pkl")

# ─── Data Augmentation & Tricks ─────────────────────────────────────────────

def spec_augment(mel_spec, freq_mask_param=15, time_mask_param=20):
    """Frequency and Time masking for a batch of spectrograms."""
    batch_size, channels, n_mels, n_frames = mel_spec.size()
    mel_spec = mel_spec.clone() # avoid modifying original tensor inplace if needed
    for i in range(batch_size):
        # Frequency masking
        f = int(np.random.uniform(0, freq_mask_param))
        if f > 0:
            f0 = int(np.random.uniform(0, n_mels - f))
            mel_spec[i, :, f0:f0+f, :] = 0
            
        # Time masking
        t = int(np.random.uniform(0, time_mask_param))
        if t > 0:
            t0 = int(np.random.uniform(0, n_frames - t))
            mel_spec[i, :, :, t0:t0+t] = 0
    return mel_spec

def mixup_data(x, y, alpha=0.2):
    """Mixup data augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_class_weights(train_loader):
    """Compute balanced class weights for the current training fold."""
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    class_weights = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=all_labels)
    return torch.FloatTensor(class_weights)

# ─── Checkpointing ───────────────────────────────────────────────────────────

def save_checkpoint(fold_accuracies, all_preds, all_labels, all_histories, last_fold):
    """Save progress so training can resume after interruption."""
    checkpoint = {
        "fold_accuracies": fold_accuracies,
        "all_preds": all_preds,
        "all_labels": all_labels,
        "all_histories": all_histories,
        "last_completed_fold": last_fold,
    }
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"  [SAVED] Checkpoint after fold {last_fold}")


def load_checkpoint():
    """Load checkpoint if it exists. Returns None if no checkpoint."""
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "rb") as f:
            checkpoint = pickle.load(f)
        print(f"  [RESUME] Found checkpoint — folds 1-{checkpoint['last_completed_fold']} already done")
        return checkpoint
    return None

# ─── Training Core ───────────────────────────────────────────────────────────

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, apply_aug=True):
    """Train for one epoch with optional SpecAugment and Mixup, plus Gradient Clipping."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        if apply_aug:
            # Apply SpecAugment
            inputs = spec_augment(inputs)
            # Apply Mixup
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=0.2)
        else:
            targets_a, targets_b, lam = labels, labels, 1.0

        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step() # OneCycleLR steps per batch

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        # Approximate mixup accuracy (compare to majority label)
        true_labels = targets_a if lam > 0.5 else targets_b
        correct += predicted.eq(true_labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """Evaluate model, return loss, accuracy, all predictions and labels."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    # Use standard CrossEntropy without weights for validation loss
    eval_criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = eval_criterion(outputs, labels)
            
            probs = torch.softmax(outputs, dim=1)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels), np.array(all_probs)


def train_single_fold(test_fold, device=DEVICE):
    """Train and evaluate on a single fold. Returns test accuracy and predictions."""
    print(f"\n{'='*60}")
    print(f"  FOLD {test_fold} / {NUM_FOLDS}")
    print(f"{'='*60}")

    # Data
    train_loader, test_loader = get_fold_dataloaders(test_fold)

    # Class Weights for the Loss Function
    class_weights = get_class_weights(train_loader).to(device)

    # Model
    model = UrbanSoundCNN(num_classes=NUM_CLASSES).to(device)
    
    # Label Smoothing + Class Weights
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # OneCycleLR Scheduler (Warmup + Cosine Decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=NUM_EPOCHS,
        pct_start=0.1 # Warmup for first 10% of training
    )

    # Training loop with Smoothed Early Stopping
    best_val_loss = float('inf')
    best_acc = 0.0
    patience = 8
    patience_counter = 0
    val_loss_history = []
    
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        # Anneal off augmentation in the last 20% of epochs (e.g., epoch 40-50)
        apply_aug = True if epoch <= (NUM_EPOCHS * 0.8) else False
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, apply_aug=apply_aug)
        test_loss, test_acc, _, _, _ = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        # Update best accuracy and save model based on raw test_acc
        if test_acc > best_acc:
            best_acc = test_acc
            fold_dir = os.path.join(RESULTS_DIR, f"fold{test_fold}")
            os.makedirs(fold_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(fold_dir, "best_model.pth"))

        # Early stopping based on 3-epoch moving average of Validation Loss
        val_loss_history.append(test_loss)
        if len(val_loss_history) >= 3:
            smoothed_val_loss = np.mean(val_loss_history[-3:])
        else:
            smoothed_val_loss = test_loss

        if smoothed_val_loss < best_val_loss:
            best_val_loss = smoothed_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            aug_status = "ON" if apply_aug else "OFF"
            print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} (Aug: {aug_status}) | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
                  f"Test Loss: {test_loss:.4f} Acc: {test_acc:.1f}% | "
                  f"Best Acc: {best_acc:.1f}% (Patience: {patience_counter}/{patience})")
            
        if patience_counter >= patience:
            print(f"  [EARLY STOPPING] Triggered at epoch {epoch}. No improvement in smoothed val loss for {patience} epochs.")
            break

    # Load best model and get final evaluation
    model.load_state_dict(torch.load(
        os.path.join(RESULTS_DIR, f"fold{test_fold}", "best_model.pth"),
        weights_only=True
    ))
    _, final_acc, all_preds, all_labels, _ = evaluate(model, test_loader, criterion, device)

    print(f"\n  Fold {test_fold} Final Best Test Accuracy: {final_acc:.2f}%")

    # Save training curves
    history = {
        "train_loss": train_losses,
        "test_loss": test_losses,
        "train_acc": train_accs,
        "test_acc": test_accs,
        "best_acc": best_acc
    }

    return final_acc, all_preds, all_labels, history

# ─── Plotting ───────────────────────────────────────────────────────────────

def plot_training_curves(all_histories, save_dir):
    """Plot training/test curves for all folds."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for fold_idx, history in enumerate(all_histories):
        fold_num = fold_idx + 1
        epochs = range(1, len(history["train_loss"]) + 1)

        axes[0].plot(epochs, history["test_loss"], alpha=0.5, label=f"Fold {fold_num}")
        axes[1].plot(epochs, history["test_acc"], alpha=0.5, label=f"Fold {fold_num}")

    axes[0].set_title("Test Loss per Fold", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8, loc='upper right', ncol=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Test Accuracy per Fold", fontsize=14)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(fontsize=8, loc='lower right', ncol=2)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"Training curves saved.")


def plot_confusion_matrix(all_preds, all_labels, save_dir):
    """Plot aggregate confusion matrix across all folds."""
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=14)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1])
    axes[1].set_title("Confusion Matrix (Normalized)", fontsize=14)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"Confusion matrix saved.")

# ─── Main Execution ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  CENTRALIZED BASELINE — 10-Fold Cross-Validation")
    print("  (With SpecAugment, Mixup, OneCycleLR, Smoothed Early Stopping)")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # Model info
    model_tmp = UrbanSoundCNN()
    total_params, trainable_params = count_parameters(model_tmp)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    del model_tmp

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Check for checkpoint (resume support)
    checkpoint = load_checkpoint()

    if checkpoint:
        fold_accuracies = checkpoint["fold_accuracies"]
        all_preds_combined = checkpoint["all_preds"]
        all_labels_combined = checkpoint["all_labels"]
        all_histories = checkpoint["all_histories"]
        start_fold = checkpoint["last_completed_fold"] + 1
        print(f"  Resuming from fold {start_fold}...")
    else:
        fold_accuracies = []
        all_preds_combined = []
        all_labels_combined = []
        all_histories = []
        start_fold = 1

    start_time = time.time()

    # 10-fold cross-validation
    for fold in range(start_fold, NUM_FOLDS + 1):
        acc, preds, labels, history = train_single_fold(fold)
        fold_accuracies.append(acc)
        all_preds_combined.extend(preds.tolist())
        all_labels_combined.extend(labels.tolist())
        all_histories.append(history)

        # Save checkpoint after each fold
        save_checkpoint(fold_accuracies, all_preds_combined, all_labels_combined,
                        all_histories, fold)

    total_time = time.time() - start_time

    # ─── Results Summary ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  10-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 60)

    for i, acc in enumerate(fold_accuracies):
        print(f"  Fold {i+1:2d}: {acc:.2f}%")

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print(f"\n  Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"  Total training time: {total_time/60:.1f} minutes")
    
    # Classification report
    all_preds_arr = np.array(all_preds_combined)
    all_labels_arr = np.array(all_labels_combined)

    report = classification_report(
        all_labels_arr, all_preds_arr,
        target_names=CLASS_NAMES, digits=3
    )
    print(f"\nClassification Report (aggregated across all folds):\n{report}")

    # Save results
    results = {
        "fold_accuracies": fold_accuracies,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "total_time_seconds": total_time,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": 32,
        "model_params": total_params,
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save classification report
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(f"10-Fold CV Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%\n")
        f.write("\n")
        f.write(report)

    # Plots
    plot_training_curves(all_histories, RESULTS_DIR)
    plot_confusion_matrix(all_preds_arr, all_labels_arr, RESULTS_DIR)

    # Clean up checkpoint (training complete)
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("Checkpoint removed (training complete).")

    print(f"\nAll results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

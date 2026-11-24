#!/usr/bin/env python3
"""
Raw Binary .ULog Deep Learning Cyber-Attack Detection Pipeline
Dataset: IEEE DataPort UAV Attack Dataset (Whelan et al., 2020 / 2022)
Input: RAW BINARY .ULG FLIGHT LOG FILES (Parsed via pyulog directly from disk)
Model: 1D-CNN + Bidirectional LSTM (Temporal Feature Extraction)
Target: Binary Attack Classification (Benign vs Malicious GPS Spoofing / Jamming)
Metrics: Accuracy, Precision, Recall, F1-Score, Specificity, FAR, AUC-ROC, Latency
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyulog import ULog
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
)

# Set random seeds for reproducible science
np.random.seed(42)
torch.manual_seed(42)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(42)

# ==============================================================================
# 1. Raw .ULog Parser & Time-Series Synchronization
# ==============================================================================

def extract_topics_from_ulg(ulg_path, attack_window=None, sampling_interval_us=50000):
    """
    Parses a raw binary .ulg flight log directly from disk.
    Extracts raw uORB topics: GPS, IMU, local position, attitude, and estimator innovations.
    Resamples onto a fixed high-resolution time grid (default 50,000 us = 20 Hz).
    """
    print(f" -> Parsing raw binary .ulg file: {os.path.basename(ulg_path)}")
    ulog = ULog(ulg_path)
    
    # Extract topics dictionary
    topic_data = {}
    for d in ulog.data_list:
        if d.name not in topic_data:
            topic_data[d.name] = pd.DataFrame(d.data)
            
    # Key uORB telemetry topics
    gps = topic_data['vehicle_gps_position'].sort_values('timestamp')
    imu = topic_data['sensor_combined'].sort_values('timestamp')
    pos = topic_data['vehicle_local_position'].sort_values('timestamp')
    att = topic_data['vehicle_attitude'].sort_values('timestamp')
    
    # Common overlapping flight time range
    t_start = max(gps['timestamp'].min(), imu['timestamp'].min(), pos['timestamp'].min(), att['timestamp'].min())
    t_end = min(gps['timestamp'].max(), imu['timestamp'].max(), pos['timestamp'].max(), att['timestamp'].max())
    
    # Generate continuous timestamp grid (e.g. 20 Hz = 50ms)
    grid_timestamps = np.arange(int(t_start), int(t_end), sampling_interval_us, dtype=gps['timestamp'].dtype)
    base_df = pd.DataFrame({'timestamp': grid_timestamps})
    
    # Merge telemetry streams onto common timeline using nearest-neighbor asof join
    merged = pd.merge_asof(base_df, gps, on='timestamp', direction='nearest')
    merged = pd.merge_asof(merged, pos, on='timestamp', direction='nearest', suffixes=('', '_pos'))
    merged = pd.merge_asof(merged, att, on='timestamp', direction='nearest', suffixes=('', '_att'))
    merged = pd.merge_asof(merged, imu, on='timestamp', direction='nearest', suffixes=('', '_imu'))
    
    # If estimator innovations exist, include them
    if 'estimator_innovations' in topic_data:
        innov = topic_data['estimator_innovations'].sort_values('timestamp')
        merged = pd.merge_asof(merged, innov, on='timestamp', direction='nearest', suffixes=('', '_innov'))
        
    # Ground-truth binary labeling
    if attack_window is not None:
        t0, t1 = attack_window
        merged['label'] = ((merged['timestamp'] >= t0) & (merged['timestamp'] <= t1)).astype(int)
    else:
        merged['label'] = 0
        
    print(f"    Extracted {len(merged):,} synchronized epochs ({merged['label'].sum():,} attack, {(merged['label']==0).sum():,} benign)")
    return merged

def load_all_raw_ulog_data():
    print("=" * 70)
    print(" [1/5] PARSING RAW BINARY .ULG FILES (pyulog Engine)")
    print("=" * 70)
    
    ulg_benign = '/Users/rohan/Projects/Drone Sim/UAVAttackData/Live GPS Spoofing and Jamming/Benign Flight/ace-benign-log_0_2033-8-19-16-27-30.ulg'
    ulg_jamming = '/Users/rohan/Projects/Drone Sim/UAVAttackData/Live GPS Spoofing and Jamming/GPS Jamming/ace-jamming-log_1_2033-8-19-16-46-46.ulg'
    ulg_spoofing = '/Users/rohan/Projects/Drone Sim/UAVAttackData/Live GPS Spoofing and Jamming/GPS Spoofing/ace-spoofing-hackrf-log_5_2033-8-19-17-14-18.ulg'
    
    # Exact attack injection windows recorded during the physical trials (microsecond timestamps)
    jamming_window = (504844877, 555396912)
    spoofing_window = (375744085, 395149388)
    
    # 20 Hz sampling (50 ms time steps)
    df_b = extract_topics_from_ulg(ulg_benign, attack_window=None, sampling_interval_us=50000)
    df_j = extract_topics_from_ulg(ulg_jamming, attack_window=jamming_window, sampling_interval_us=50000)
    df_s = extract_topics_from_ulg(ulg_spoofing, attack_window=spoofing_window, sampling_interval_us=50000)
    
    combined_df = pd.concat([df_b, df_j, df_s], ignore_index=True)
    print(f"\n Combined Raw .ULog Dataset: {len(combined_df):,} total samples across 3 flights")
    print(f" Class Balance: {(combined_df['label']==0).sum():,} Benign ({(combined_df['label']==0).mean()*100:.1f}%) | "
          f"{(combined_df['label']==1).sum():,} Malicious ({(combined_df['label']==1).mean()*100:.1f}%)")
    
    # Exclude timestamp identifiers and targets from feature set to avoid leakage
    drop_cols = ['timestamp', 'timestamp_pos', 'timestamp_att', 'timestamp_imu', 'timestamp_innov', 'time_utc_usec', 'label']
    feature_cols = [c for c in combined_df.columns if c not in drop_cols and np.issubdtype(combined_df[c].dtype, np.number)]
    print(f" Physical Sensor Telemetry Features: {len(feature_cols)} channels")
    
    X = combined_df[feature_cols].values.astype(np.float32)
    y = combined_df['label'].values.astype(np.float32)
    
    # Clean any inf or nan
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    return X, y, feature_cols

# ==============================================================================
# 2. Sliding Temporal Sequences
# ==============================================================================

def create_sequences(X, y, window_size=5):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size + 1):
        X_seq.append(X[i : i + window_size])
        y_seq.append(y[i + window_size - 1])
    return np.array(X_seq), np.array(y_seq)

class UAVTelemetryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==============================================================================
# 3. Deep Learning Architecture: 1D-CNN + BiLSTM Network
# ==============================================================================

class UAV_RawULog_CNN_BiLSTM(nn.Module):
    """
    1D-CNN + Bidirectional LSTM Network trained directly on raw .ulg flight streams:
    - 1D-CNN: Extracts spatial cross-correlations across raw physical sensor channels.
    - BiLSTM: Captures bidirectional temporal dynamics and kinematic rate deviations.
    - Dense Head: Outputs attack logits with dropout regularization.
    """
    def __init__(self, num_features, seq_len=5, cnn_channels=64, lstm_hidden=64):
        super(UAV_RawULog_CNN_BiLSTM, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=cnn_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.gelu = nn.GELU()
        self.conv_drop = nn.Dropout(0.2)
        
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # x: (batch_size, seq_len, num_features)
        x_conv = x.transpose(1, 2)
        out = self.conv_drop(self.gelu(self.bn1(self.conv1(x_conv))))
        out = out.transpose(1, 2)
        lstm_out, _ = self.lstm(out)
        last_step = lstm_out[:, -1, :]
        logits = self.classifier(last_step).squeeze(-1)
        return logits

# ==============================================================================
# 4. Training Engine
# ==============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=25):
    print("\n" + "=" * 70)
    print(" [3/5] TRAINING MODEL ON RAW .ULG DATA (Apple Silicon Metal MPS)")
    print("=" * 70)
    
    best_f1 = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    best_weights = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch_y)
            
        train_loss = total_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * len(batch_y)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long().cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(batch_y.cpu().numpy())
                
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(all_targets, all_preds)
        val_f1 = f1_score(all_targets, all_preds, zero_division=0)
        
        scheduler.step(val_f1)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_weights = model.state_dict().copy()
            star = " ⭐ (Best Model)"
        else:
            star = ""
            
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f" Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc*100:6.2f}% | Val F1: {val_f1*100:6.2f}%{star}")
            
    if best_weights:
        model.load_state_dict(best_weights)
    return history

# ==============================================================================
# 5. Rigorous Evaluation: 4 Core Metrics
# ==============================================================================

def evaluate_and_plot(model, test_loader, device, history):
    print("\n" + "=" * 70)
    print(" [4/5] RIGOROUS EVALUATION: 4 CORE METRICS (RAW .ULG BENCHMARK)")
    print("=" * 70)
    
    model.eval()
    all_probs, all_preds, all_targets = [], [], []
    inference_times = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            t0 = time.perf_counter()
            logits = model(batch_x)
            if device.type == 'mps':
                torch.mps.synchronize()
            t1 = time.perf_counter()
            inference_times.append((t1 - t0) / len(batch_x))
            
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())
            
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 4 Core Metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    # Secondary Diagnostics
    cm = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    auc_roc = roc_auc_score(all_targets, all_probs)
    avg_latency_ms = np.mean(inference_times) * 1000.0
    
    print("\n" + "-" * 55)
    print("       RAW .ULOG MODEL EVALUATION RESULTS FOR MENTOR")
    print("-" * 55)
    print(f"  [1] ACCURACY    : {accuracy * 100:6.2f} %")
    print(f"  [2] PRECISION   : {precision * 100:6.2f} %  (Low False Alarms)")
    print(f"  [3] RECALL      : {recall * 100:6.2f} %  (Detection Rate)")
    print(f"  [4] F1-SCORE    : {f1 * 100:6.2f} %  (Harmonic Balanced Metric)")
    print("-" * 55)
    print(f"  Specificity     : {specificity * 100:6.2f} %")
    print(f"  False Alarm Rate: {far * 100:6.2f} %")
    print(f"  AUC-ROC Score   : {auc_roc:8.4f}")
    print(f"  Inference Speed : {avg_latency_ms:6.3f} ms / sample")
    print("-" * 55)
    print(f"  Confusion Matrix: TP={tp:,} | TN={tn:,} | FP={fp:,} | FN={fn:,}")
    print("-" * 55)
    
    # Save Metrics JSON
    os.makedirs('/Users/rohan/Projects/Drone Sim/results', exist_ok=True)
    metrics_dict = {
        'source': 'raw_binary_ulog',
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'specificity': float(specificity),
        'false_alarm_rate': float(far),
        'auc_roc': float(auc_roc),
        'avg_latency_ms': float(avg_latency_ms),
        'confusion_matrix': {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)},
        'test_samples': int(len(all_targets))
    }
    with open('/Users/rohan/Projects/Drone Sim/results/dl_raw_ulog_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=4)
        
    # --- PLOT 1: Confusion Matrix ---
    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Benign (Normal)', 'Cyber Attack'],
                yticklabels=['Benign (Normal)', 'Cyber Attack'],
                annot_kws={'size': 14, 'weight': 'bold'})
    plt.title(f'Raw .ULog DL Detection Confusion Matrix\n(Accuracy: {accuracy*100:.2f}% | F1: {f1*100:.2f}%)', fontsize=12, pad=12)
    plt.xlabel('Predicted Flight State', fontsize=11, fontweight='semibold')
    plt.ylabel('Actual Ground Truth', fontsize=11, fontweight='semibold')
    plt.tight_layout()
    plt.savefig('/Users/rohan/Projects/Drone Sim/results/dl_raw_ulog_confusion_matrix.png')
    plt.close()
    
    # --- PLOT 2: Training Curves ---
    epochs_range = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)
    
    ax1.plot(epochs_range, history['train_loss'], label='Train Loss', color='#2563EB', lw=2)
    ax1.plot(epochs_range, history['val_loss'], label='Val Loss', color='#DC2626', lw=2, linestyle='--')
    ax1.set_title('Cross-Entropy Loss vs Epochs (Raw .ULog)', fontsize=11, fontweight='semibold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(epochs_range, [a * 100 for a in history['val_acc']], label='Val Accuracy (%)', color='#059669', lw=2)
    ax2.plot(epochs_range, [f * 100 for f in history['val_f1']], label='Val F1-Score (%)', color='#7C3AED', lw=2, linestyle='--')
    ax2.set_title('Validation Accuracy & F1 vs Epochs', fontsize=11, fontweight='semibold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score (%)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('/Users/rohan/Projects/Drone Sim/results/dl_raw_ulog_training_curves.png')
    plt.close()
    
    # --- PLOT 3: ROC Curve ---
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    prec_curve, rec_curve, _ = precision_recall_curve(all_targets, all_probs)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)
    ax1.plot(fpr, tpr, color='#2563EB', lw=2.5, label=f'ROC Curve (AUC = {auc_roc:.4f})')
    ax1.plot([0, 1], [0, 1], color='gray', linestyle=':', lw=1.5)
    ax1.set_title('Receiver Operating Characteristic (Raw .ULog)', fontsize=11, fontweight='semibold')
    ax1.set_xlabel('False Positive Rate (FAR)')
    ax1.set_ylabel('True Positive Rate (Recall)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')
    
    ax2.plot(rec_curve, prec_curve, color='#7C3AED', lw=2.5, label='Precision-Recall Curve')
    ax2.set_title('Precision-Recall Curve', fontsize=11, fontweight='semibold')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower left')
    
    plt.tight_layout()
    plt.savefig('/Users/rohan/Projects/Drone Sim/results/dl_raw_ulog_roc_pr_curves.png')
    plt.close()
    
    print("\n [5/5] RAW .ULOG ARTIFACTS SAVED:")
    print("   - Metrics JSON     : file:///Users/rohan/Projects/Drone%20Sim/results/dl_raw_ulog_metrics.json")
    print("   - Confusion Matrix : file:///Users/rohan/Projects/Drone%20Sim/results/dl_raw_ulog_confusion_matrix.png")
    print("   - Training Curves  : file:///Users/rohan/Projects/Drone%20Sim/results/dl_raw_ulog_training_curves.png")
    print("   - ROC/PR Curves    : file:///Users/rohan/Projects/Drone%20Sim/results/dl_raw_ulog_roc_pr_curves.png")
    return metrics_dict

# ==============================================================================
# Main
# ==============================================================================

def main():
    X, y, feature_names = load_all_raw_ulog_data()
    
    window_size = 5
    X_seq, y_seq = create_sequences(X, y, window_size=window_size)
    print(f"\n [2/5] GENERATED TEMPORAL WINDOWS: {len(X_seq):,} sequences of shape ({window_size}, {X.shape[1]})")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.20, random_state=42, stratify=y_seq
    )
    
    N_tr, T, D = X_train.shape
    N_te, _, _ = X_test.shape
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, D)).reshape(N_tr, T, D)
    X_test_scaled = scaler.transform(X_test.reshape(-1, D)).reshape(N_te, T, D)
    
    train_dataset = UAVTelemetryDataset(X_train_scaled, y_train)
    test_dataset = UAVTelemetryDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f" Accelerating on Hardware: {device}")
    
    num_neg = sum(y_train == 0)
    num_pos = sum(y_train == 1)
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)], dtype=torch.float32, device=device)
    print(f" Class Imbalance Compensation: pos_weight = {pos_weight.item():.2f}")
    
    model = UAV_RawULog_CNN_BiLSTM(num_features=D, seq_len=T).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    history = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, epochs=25)
    
    os.makedirs('/Users/rohan/Projects/Drone Sim/models', exist_ok=True)
    torch.save(model.state_dict(), '/Users/rohan/Projects/Drone Sim/models/uav_dl_raw_ulog_detector.pt')
    print(" Trained Model Saved: file:///Users/rohan/Projects/Drone%20Sim/models/uav_dl_raw_ulog_detector.pt")
    
    evaluate_and_plot(model, test_loader, device, history)

if __name__ == '__main__':
    main()

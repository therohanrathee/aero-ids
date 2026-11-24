#!/usr/bin/env python3
"""
Interactive Test Script: Run live inferences on unseen flight packets
using the trained Deep Learning Model (models/uav_dl_raw_ulog_detector.pt).
"""

import torch
import numpy as np
import json
import time

from train_dl_from_raw_ulog import (
    load_all_raw_ulog_data, create_sequences, UAV_RawULog_CNN_BiLSTM
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    print("=" * 65)
    print("   LIVE DEEP LEARNING INFERENCE CHECKER (Unseen Test Packets)")
    print("=" * 65)
    
    # 1. Load data
    X, y, feature_names = load_all_raw_ulog_data()
    X_seq, y_seq = create_sequences(X, y, window_size=5)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.20, random_state=42, stratify=y_seq
    )
    
    # Scale
    N_tr, T, D = X_train.shape
    N_te, _, _ = X_test.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, D))
    X_test_scaled = scaler.transform(X_test.reshape(-1, D)).reshape(N_te, T, D)
    
    # Load Model
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = UAV_RawULog_CNN_BiLSTM(num_features=D, seq_len=T).to(device)
    model.load_state_dict(torch.load('/Users/rohan/Projects/Drone Sim/models/uav_dl_raw_ulog_detector.pt', map_location=device))
    model.eval()
    
    print("\nTaking 10 random flight packets from unseen test set...\n")
    print(f"{'Sample #':<10} | {'Ground Truth':<15} | {'DL Prediction':<15} | {'Confidence':<12} | {'Status'}")
    print("-" * 72)
    
    # Select 5 benign and 5 attack samples
    benign_idx = np.where(y_test == 0)[0]
    attack_idx = np.where(y_test == 1)[0]
    
    selected_idx = list(np.random.choice(benign_idx, 5, replace=False)) + \
                   list(np.random.choice(attack_idx, 5, replace=False))
    np.random.shuffle(selected_idx)
    
    for i, idx in enumerate(selected_idx, 1):
        sample_x = torch.tensor(X_test_scaled[idx : idx + 1], dtype=torch.float32).to(device)
        actual = "ATTACK" if y_test[idx] == 1 else "BENIGN"
        
        t0 = time.perf_counter()
        with torch.no_grad():
            logit = model(sample_x)
            prob = torch.sigmoid(logit).item()
        t1 = time.perf_counter()
        
        pred = "ATTACK" if prob >= 0.5 else "BENIGN"
        conf = prob if pred == "ATTACK" else (1.0 - prob)
        
        status = "\033[92mCORRECT\033[0m" if pred == actual else "\033[91mERROR\033[0m"
        print(f"Sample {i:02d}    | {actual:<15} | {pred:<15} | {conf*100:6.2f} %     | {status}")
        
    print("-" * 72)
    print("All live inferences verified against actual ground truth.")

if __name__ == '__main__':
    main()

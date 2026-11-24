"""
Script to display a side-by-side comparison of Raw ULog binary vs. Processed Telemetry Data.
Designed for UAV Flight Cyber-Physical Defense project.
"""

import os
import pyulog
import pandas as pd
import numpy as np

RAW_ULOG_PATH = "UAVAttackData/Live GPS Spoofing and Jamming/GPS Spoofing/ace-spoofing-hackrf-log_5_2033-8-19-17-14-18.ulg"
PROCESSED_CSV_PATH = "UAVAttackData/Live GPS Spoofing and Jamming/GPS Spoofing/Processed/spoofing-merged-gps-only.csv"

def display_raw_ulog(path):
    print("=" * 80)
    print(" 1. RAW ULOG FILE ON DISK (BINARY HEX DUMP)")
    print("=" * 80)
    print(f"File Path: {path}")
    file_size_kb = os.path.getsize(path) / 1024
    print(f"File Size: {file_size_kb:.2f} KB\n")
    
    with open(path, "rb") as f:
        raw_bytes = f.read(96)
        
    print("First 96 Bytes (Magic Header + Topic Definitions):")
    print("-" * 75)
    for i in range(0, len(raw_bytes), 16):
        chunk = raw_bytes[i:i+16]
        hex_str = " ".join(f"{b:02x}" for b in chunk)
        ascii_str = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        print(f"{i:08x}:  {hex_str:<48}  |{ascii_str}|")
    print("-" * 75)
    print("Notice the magic bytes: '55 4c 6f 67 01 12' -> ASCII 'ULog\\x01\\x12'")
    print("This is PX4's binary serialization format (not human-readable text).\n")

    print("=" * 80)
    print(" 2. PARSED uORB TOPICS INSIDE THE RAW ULOG (Unsynchronized Multi-Rate Streams)")
    print("=" * 80)
    ulog = pyulog.ULog(path)
    dur_s = (ulog.last_timestamp - ulog.start_timestamp) / 1e6
    print(f"Flight Duration: {dur_s:.2f} seconds ({dur_s/60:.2f} minutes)")
    print(f"Total Logged Topics: {len(ulog.data_list)}\n")
    
    # Show sampling rate mismatch between topics
    sample_topics = ["sensor_combined", "vehicle_attitude", "vehicle_local_position", "vehicle_gps_position"]
    print(f"{'Topic Name':<28} | {'Total Samples':<14} | {'Approx Frequency':<18} | {'Sample Timestamp (us)':<22}")
    print("-" * 90)
    for name in sample_topics:
        matches = [d for d in ulog.data_list if d.name == name]
        if matches:
            d = matches[0]
            n_samples = len(d.data["timestamp"])
            freq = n_samples / dur_s
            t0 = d.data["timestamp"][0]
            print(f"{name:<28} | {n_samples:<14} | {freq:>6.1f} Hz          | {t0:<22}")
    print("-" * 90)
    print(">> Note the problem: IMU logs at 200 Hz (27,271 samples), but GPS logs at only 1 Hz (138 samples)!")
    print(">> Their timestamps DO NOT align. That is why raw ULog cannot be fed directly to a DL model.\n")

def display_processed_ulog(path):
    print("=" * 80)
    print(" 3. PROCESSED & SYNCHRONIZED TELEMETRY (What the DL Model Consumes)")
    print("=" * 80)
    print(f"Authors' Pre-Processed CSV: {path}")
    df_csv = pd.read_csv(path)
    print(f"Shape: {df_csv.shape[0]} rows x {df_csv.shape[1]} columns\n")
    
    key_cols = [
        "timestamp", "lat_x", "lon_x", "alt_x", 
        "vx", "vy", "vz", 
        "q[0]", "q[1]", "q[2]", "q[3]"
    ]
    existing_cols = [c for c in key_cols if c in df_csv.columns]
    
    print("Sample Rows from Processed File (Aligned on unified timestamps):")
    print(df_csv[existing_cols].head(5).to_string(index=False))
    print("\n" + "=" * 80)

if __name__ == "__main__":
    display_raw_ulog(RAW_ULOG_PATH)
    display_processed_ulog(PROCESSED_CSV_PATH)

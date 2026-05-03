"""
Cantilever Beam SHM — Merge and Verify All Sessions
=====================================================
Run this AFTER collecting all 12 sessions (4 levels x 3 sessions).

What it does:
    1. Loads all damage_L{n}_S{n}.csv files
    2. Merges into one master dataset
    3. Verifies actual sampling rate per session
    4. Plots raw waveform per damage level to sanity check
    5. Saves: cantilever_all_sessions.csv

Run in Jupyter:
    %run merge_sessions.py
Or:
    exec(open('merge_sessions.py').read())
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
# ── Load all session files ────────────────────────────────────────────────────
files = []
for level in range(4):
    for session in range(1, 4):
        fname = f"damage_L{level}_S{session}.csv"
        if os.path.exists(fname):
            df = pd.read_csv(fname)
            files.append(df)
            print(f"Loaded {fname}: {len(df):,} rows")
        else:
            print(f"MISSING: {fname} — collect this session first")

if not files:
    print("No session files found. Run collect_data.py first.")
    raise SystemExit

# ── Merge ─────────────────────────────────────────────────────────────────────
df_all = pd.concat(files, ignore_index=True)

# Re-assign sample_id globally (monotonic across all sessions)
df_all['global_sample_id'] = range(len(df_all))

print(f"\nTotal rows: {len(df_all):,}")
print(f"Damage level distribution:")
print(df_all['damage_label'].value_counts())

# ── Verify actual sampling rate per session ───────────────────────────────────
print("\n-- Sampling Rate Check -----------------------------------------")
print(f"{'Session':<20} {'Rows':>8} {'Expected Hz':>12} {'Notes'}")
print("-" * 60)

for label, group in df_all.groupby('session'):
    rows = len(group)
    # Each session should be ~120s at 100Hz = ~12000 rows
    expected = 12000
    rate_pct  = 100 * rows / expected
    status = "OK" if rate_pct > 90 else "LOW — check baud rate"
    print(f"{label:<20} {rows:>8,} {rate_pct:>11.1f}% {status}")

# ── Plot raw az_zeroed per damage level ───────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=False)
colors = ['green', 'goldenrod', 'darkorange', 'red']
labels = ['Level 0 — Healthy', 'Level 1 — Light', 'Level 2 — Moderate', 'Level 3 — Severe']

for i, (level, color, label) in enumerate(zip(range(4), colors, labels)):
    subset = df_all[df_all['damage_level'] == level].head(3000)  # first 30s
    axes[i].plot(subset['global_sample_id'], subset['az_zeroed'],
                 linewidth=0.6, color=color, alpha=0.8)
    axes[i].set_ylabel('az_zeroed', fontsize=10)
    axes[i].set_title(label, fontsize=11, fontweight='bold')
    axes[i].grid(True, alpha=0.3)

axes[-1].set_xlabel('Sample ID', fontsize=11)
plt.suptitle('Raw Accelerometer Signal per Damage Level\n(First 30 seconds of Session 1)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('raw_waveforms.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved: raw_waveforms.png")

# ── Check for clipping ────────────────────────────────────────────────────────
print("\n── Clipping Check (values at ±32767 = sensor overflow) ─────────────")
for level in range(4):
    subset = df_all[df_all['damage_level'] == level]
    clipped = ((subset['az'] == 32767) | (subset['az'] == -32768)).sum()
    print(f"  Level {level}: {clipped} clipped rows out of {len(subset):,}")
    if clipped > 0:
        print(f"    WARNING: Clipping detected — consider switching to ±4g range in Arduino")

# ── Save master dataset ───────────────────────────────────────────────────────
df_all.to_csv('cantilever_all_sessions.csv', index=False)
print(f"\nSaved: cantilever_all_sessions.csv ({len(df_all):,} rows)")
print("Next: run feature_extraction.py")
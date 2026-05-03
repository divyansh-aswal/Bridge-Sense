"""
Cantilever Beam SHM — Feature Extraction + ML Training
========================================================
Run after merge_sessions.py

Input:  cantilever_all_sessions.csv
Output: cantilever_features.csv
        confusion_matrix.png
        feature_importance.png
        health_index.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ── CONFIG ────────────────────────────────────────────────────────────────────
SAMPLING_RATE = 100   # Hz
WINDOW        = 200   # samples = 2 seconds per window
OVERLAP       = 100   # 50% overlap between windows

DAMAGE_LABELS = {
    0: 'Healthy',
    1: 'Light Damage',
    2: 'Moderate Damage',
    3: 'Severe Damage'
}
COLORS = {0: 'green', 1: 'goldenrod', 2: 'darkorange', 3: 'red'}

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('cantilever_all_sessions.csv')
print(f"  {len(df):,} rows loaded")
print(f"  Damage level counts: {dict(df['damage_level'].value_counts().sort_index())}")

# ── 2. Extract features ───────────────────────────────────────────────────────
print(f"\nExtracting features...")
print(f"  Window: {WINDOW} samples = {WINDOW/SAMPLING_RATE:.1f}s")
print(f"  Overlap: {OVERLAP} samples = 50%")

features = []

for i in range(0, len(df) - WINDOW, OVERLAP):
    w_df = df.iloc[i:i+WINDOW]
    w    = w_df['az_zeroed'].values
    wx   = w_df['ax'].values
    wy   = w_df['ay'].values

    mid          = w_df.iloc[WINDOW // 2]
    damage_level = mid['damage_level']
    session      = mid['session']
    sample_id    = mid['global_sample_id']

    # Remove any DC offset within window (detrend)
    w  = w  - np.mean(w)
    wx = wx - np.mean(wx)
    wy = wy - np.mean(wy)

    # ── Time domain ──────────────────────────────────────────────
    rms          = np.sqrt(np.mean(w**2))
    std_dev      = np.std(w)
    kurt         = kurtosis(w)
    skewness     = skew(w)
    peak         = np.max(np.abs(w))
    mean_abs     = np.mean(np.abs(w))
    crest_factor = peak / (rms + 1e-10)
    shape_factor = rms / (mean_abs + 1e-10)
    energy       = np.sum(w**2)

    # ── Frequency domain ──────────────────────────────────────────
    fft_vals  = np.abs(fft(w))[:WINDOW // 2]
    freqs     = fftfreq(WINDOW, d=1.0/SAMPLING_RATE)[:WINDOW // 2]
    total_pwr = np.sum(fft_vals**2) + 1e-10

    # Skip DC bin (index 0) when finding dominant frequency
    dominant_freq = freqs[np.argmax(fft_vals[1:]) + 1]

    # Spectral centroid
    spec_centroid = np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-10)

    # Band energies — tuned for 100Hz sensor, 0-50Hz range
    # Your cantilever natural freq should be 5-25 Hz
    band_low    = np.sum(fft_vals[(freqs >= 1)  & (freqs < 8)]**2)  / total_pwr
    band_nat    = np.sum(fft_vals[(freqs >= 8)  & (freqs < 25)]**2) / total_pwr  # natural freq band
    band_high   = np.sum(fft_vals[(freqs >= 25) & (freqs < 50)]**2) / total_pwr

    # Spectral entropy
    psd_norm     = fft_vals**2 / total_pwr
    spec_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))

    # ── AR(2) model ───────────────────────────────────────────────
    try:
        X_ar = np.column_stack([w[1:-1], w[:-2]])
        y_ar = w[2:]
        ar_coeffs, _, _, _ = np.linalg.lstsq(X_ar, y_ar, rcond=None)
        ar1, ar2      = ar_coeffs
        ar_resid_std  = np.std(y_ar - X_ar @ ar_coeffs)
    except Exception:
        ar1, ar2, ar_resid_std = 0.0, 0.0, 0.0

    # ── Cross-axis correlation ─────────────────────────────────────
    corr_xz = np.corrcoef(wx, w)[0, 1] if np.std(wx) > 0 else 0
    corr_yz = np.corrcoef(wy, w)[0, 1] if np.std(wy) > 0 else 0

    features.append({
        'sample_id':      sample_id,
        'session':        session,
        'damage_level':   damage_level,
        'damage_label':   DAMAGE_LABELS[damage_level],
        # Time domain
        'RMS':            round(rms, 4),
        'Std':            round(std_dev, 4),
        'Kurtosis':       round(kurt, 4),
        'Skewness':       round(skewness, 4),
        'Peak':           round(peak, 4),
        'Crest_Factor':   round(crest_factor, 4),
        'Shape_Factor':   round(shape_factor, 4),
        'Energy':         round(energy, 4),
        # Frequency domain
        'Natural_Freq':   round(dominant_freq, 4),
        'Spec_Centroid':  round(spec_centroid, 4),
        'Band_Low':       round(band_low, 6),
        'Band_Natural':   round(band_nat, 6),
        'Band_High':      round(band_high, 6),
        'Spec_Entropy':   round(spec_entropy, 4),
        # AR model
        'AR1':            round(float(ar1), 6),
        'AR2':            round(float(ar2), 6),
        'AR_Resid_Std':   round(ar_resid_std, 4),
        # Cross-axis
        'Corr_XZ':        round(corr_xz, 4),
        'Corr_YZ':        round(corr_yz, 4),
    })

feat_df = pd.DataFrame(features).dropna()
feat_df.to_csv('cantilever_features.csv', index=False)
print(f"  {len(feat_df):,} feature windows extracted")
print(f"  Saved: cantilever_features.csv")

# ── 3. Feature statistics per damage level ────────────────────────────────────
print("\n-- Feature Means per Damage Level -----------------------------------")
key_features = ['RMS', 'Natural_Freq', 'Kurtosis', 'Crest_Factor', 'Band_Natural']
print(feat_df.groupby('damage_label')[key_features].mean().round(3).to_string())

# ── 4. Feature trend plots ────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

for level in range(4):
    subset = feat_df[feat_df['damage_level'] == level]
    c = COLORS[level]
    l = DAMAGE_LABELS[level]
    axes[0].scatter(subset['sample_id'], subset['RMS'],         color=c, label=l, s=8, alpha=0.6)
    axes[1].scatter(subset['sample_id'], subset['Natural_Freq'],color=c, label=l, s=8, alpha=0.6)
    axes[2].scatter(subset['sample_id'], subset['Kurtosis'],    color=c, label=l, s=8, alpha=0.6)

axes[0].set_ylabel('RMS',          fontsize=11)
axes[1].set_ylabel('Natural Freq (Hz)', fontsize=11)
axes[2].set_ylabel('Kurtosis',     fontsize=11)
axes[2].set_xlabel('Sample ID (damage progression →)', fontsize=11)
axes[0].set_title('Key Features across Progressive Damage', fontsize=13, fontweight='bold')
axes[0].legend(loc='upper right', markerscale=3)
for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('feature_trends.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: feature_trends.png")

# ── 5. Train Random Forest ────────────────────────────────────────────────────
print("\n-- Training Random Forest Classifier ---------------------------")

FEATURE_COLS = [
    'RMS', 'Std', 'Kurtosis', 'Crest_Factor', 'Shape_Factor', 'Energy',
    'Natural_Freq', 'Spec_Centroid', 'Band_Low', 'Band_Natural', 'Band_High',
    'Spec_Entropy', 'AR1', 'AR2', 'AR_Resid_Std', 'Corr_XZ', 'Corr_YZ'
]

X = feat_df[FEATURE_COLS].values
y = feat_df['damage_level'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc    = (y_pred == y_test).mean() * 100

print(f"\n  Test Accuracy: {acc:.1f}%")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=[DAMAGE_LABELS[i] for i in range(4)]))

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"  5-fold CV Accuracy: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

# ── 6. Confusion matrix ───────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[DAMAGE_LABELS[i] for i in range(4)],
            yticklabels=[DAMAGE_LABELS[i] for i in range(4)])
plt.title(f'Confusion Matrix — Test Accuracy: {acc:.1f}%', fontsize=13, fontweight='bold')
plt.ylabel('Actual', fontsize=11)
plt.xlabel('Predicted', fontsize=11)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: confusion_matrix.png")

# ── 7. Feature importance ──────────────────────────────────────────────────────
importances = model.feature_importances_
sorted_idx  = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 5))
bars = plt.bar(range(len(FEATURE_COLS)),
               importances[sorted_idx],
               color='steelblue', alpha=0.8)
plt.xticks(range(len(FEATURE_COLS)),
           [FEATURE_COLS[i] for i in sorted_idx],
           rotation=45, ha='right', fontsize=10)
plt.title('Feature Importance — Which features drive damage detection?',
          fontsize=13, fontweight='bold')
plt.ylabel('Importance Score', fontsize=11)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: feature_importance.png")
print(f"\n  Top feature: {FEATURE_COLS[sorted_idx[0]]} ({importances[sorted_idx[0]]*100:.1f}%)")
print(f"  2nd feature: {FEATURE_COLS[sorted_idx[1]]} ({importances[sorted_idx[1]]*100:.1f}%)")

# ── 8. Health index ───────────────────────────────────────────────────────────
probs        = model.predict_proba(X_test)
weights      = np.array([0.0, 0.33, 0.66, 1.0])
health_index = probs @ weights

# Sort by sample_id for plot
test_ids = feat_df['sample_id'].values[
    np.where(np.isin(range(len(feat_df)), 
    np.arange(len(X_test))))[0]
]
# Simpler: just use index order
hi_df = pd.DataFrame({
    'actual':       y_test,
    'health_index': health_index
}).sort_values('actual')

plt.figure(figsize=(14, 4))
scatter = plt.scatter(range(len(hi_df)), hi_df['health_index'],
                      c=hi_df['actual'], cmap='RdYlGn_r',
                      s=15, alpha=0.7, vmin=0, vmax=3)
plt.axhline(0.25, color='orange', linestyle='--', linewidth=1.5, label='Warning (0.25)')
plt.axhline(0.55, color='red',    linestyle='--', linewidth=1.5, label='Critical (0.55)')
plt.colorbar(scatter, label='Actual Damage Level')
plt.legend(fontsize=10)
plt.ylabel('Health Index (0=Healthy → 1=Severe)', fontsize=11)
plt.xlabel('Test samples (sorted by damage level)', fontsize=11)
plt.title('Health Index — Continuous Damage Score', fontsize=13, fontweight='bold')
plt.ylim(-0.05, 1.1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('health_index.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: health_index.png")

print("\n" + "="*50)
print("  ALL DONE")
print("="*50)
print(f"  Features file   : cantilever_features.csv")
print(f"  Test accuracy   : {acc:.1f}%")
print(f"  CV accuracy     : {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")
print(f"  Plots saved     : feature_trends, confusion_matrix,")
print(f"                    feature_importance, health_index")
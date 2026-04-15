"""
Full Kalman+tracking pipeline evaluation with the trained 5-class CNN.
Runs on office1_0 and reports precision/recall/F1 (6-class: 0=absence + 5 postures).

Pipeline: HeatDetection → KalmanBlobTracker → ResidualClassification
        → 5-class CNN posture classifier (only when human detected)
        → label=0 if no human blob detected
Saves: output/case_study/pipeline_5class/predictions.json, metrics.json
"""
import sys
from pathlib import Path

WORKSPACE = Path("/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace")
SRC_DIR   = WORKSPACE / "src"
sys.path.insert(0, str(SRC_DIR))

OUTPUT_DIR = WORKSPACE / "output" / "case_study" / "pipeline_5class"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import json, torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset.dataset import ThermalDataset
from heatsource_detection_module.extract import HeatSourceDetector
from posture_detection_module.CNN_model import SimpleIRA_CNN
from posture_detection_module.utils import (
    remap_labels_simple, inverse_remap_labels_simple,
    label_to_text_simple, ThermalInvariantPreprocessor
)
from organizer_module.track_kalman import Tracker
from sklearn.metrics import (
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay, precision_recall_fscore_support
)

# ── paths ──────────────────────────────────────────────────────────────────
TEST_PATH     = WORKSPACE / "data" / "office1_0"
MODEL_PATH    = WORKSPACE / "weights" / "CNN_5class_new.pth"

# ── 5-class label mapping (for remapping) ────────────────────────────────────
# label_to_index used during training: {2:0, 3:1, 4:2, 5:3, 6:4}
# So remap_labels_simple(3) → 1, inverse_remap_labels_simple(1) → 3
RAW_5 = [2, 3, 4, 5, 6]   # the 5 posture classes
RAW_6 = [0, 2, 3, 4, 5, 6]  # full 6-class set for reporting

# ── load test dataset ───────────────────────────────────────────────────────
print("Loading test dataset (office1_0) …")
test_ds = ThermalDataset(str(TEST_PATH), noCam=True)
print(f"  Total frames: {len(test_ds)}")

# ── init pipeline components ─────────────────────────────────────────────────
heat_detector  = HeatSourceDetector()
postprocessor  = __import__('organizer_module.postprocessor', fromlist=['PostProcessor']).PostProcessor()
thermal_preproc = ThermalInvariantPreprocessor()

posture_model = SimpleIRA_CNN(num_classes=5)
posture_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
posture_model.eval()

tracker = Tracker()

# ── run inference ────────────────────────────────────────────────────────────
print("\n=== Running full pipeline on office1_0 ===")
gt_list  = []
pred_list = []

for idx in tqdm(range(len(test_ds))):
    label = test_ds.annotations_expanded[idx]
    gt_list.append(label)

    ira = test_ds.get_ira_highres(idx)
    ira[ira < 18] = 18

    # 1. Heat source detection
    thresh, mask = heat_detector.get_thresh_mask_otsu(ira)
    mask_individual = heat_detector.process_frame_connected_components(ira, min_size=100)

    # 2. Kalman tracking + residual classification
    tracker.update_blobs(mask_individual, ira,
                        heat_detector.get_unmasked_mean(ira, mask), idx)
    postprocessor.get_blobs(tracker.blobs, idx)

    # 3. Presence check: is there any non-residual blob?
    has_human = any(not blob.is_residual for blob in tracker.blobs
                     if blob.mean_temp is not None and blob.centroid is not None)

    if has_human:
        # 4. 5-class posture classification
        ira_norm = thermal_preproc(ira)                   # (62, 80) float
        # Resize to (60, 80) to match what the model was trained on
        ira_t = torch.from_numpy(ira_norm).float().unsqueeze(0).unsqueeze(0)  # (1,1,62,80)
        ira_t = torch.nn.functional.interpolate(ira_t, size=(60, 80),
                                                mode="bilinear", align_corners=False)
        ira_t = ira_t.squeeze()                            # (60, 80) — remove all size-1 dims
        posture_idx  = posture_model(ira_t.unsqueeze(0)).argmax(dim=1).item()  # (1,60,80) → scalar
        posture_raw  = inverse_remap_labels_simple(posture_idx)    # maps index → raw label
        pred_list.append(posture_raw)
    else:
        # No human detected → absence
        pred_list.append(0)

# ── save predictions ─────────────────────────────────────────────────────────
pred_path = OUTPUT_DIR / "predictions.json"
with open(pred_path, "w") as f:
    json.dump(pred_list, f, indent=2)
print(f"\nPredictions saved → {pred_path}")

# ── evaluate ─────────────────────────────────────────────────────────────────
y_gt   = gt_list
y_pred = pred_list
n = len(y_gt)
print(f"Total frames evaluated: {n}")

# --- Overall accuracy ---
overall_acc = sum(p == g for p, g in zip(y_pred, y_gt)) / n
print(f"\nOverall frame-level accuracy: {overall_acc:.4f}")

# --- Presence metrics (binary: 0=absent, 1+=present) ---
y_gt_pres   = [1 if g in [1, 2, 3, 4, 5, 6] else 0 for g in y_gt]
y_pred_pres = [1 if p in [2, 3, 4, 5, 6] else 0 for p in y_pred]
p_pres, r_pres, f1_pres, _ = precision_recall_fscore_support(
    y_gt_pres, y_pred_pres, average='binary', zero_division=0)
print(f"Presence  — Precision: {p_pres:.4f}  Recall: {r_pres:.4f}  F1: {f1_pres:.4f}")

# --- 6-class full metrics (skip GT label=1 "presence unclassified") ---
mask_valid = [(g != 1 and g != -1) for g in y_gt]
y_gt_filt   = [y_gt[i]   for i in range(n) if mask_valid[i]]
y_pred_filt = [y_pred[i] for i in range(n) if mask_valid[i]]
print(f"\nFrames used for 6-class eval (excl. GT=1/-1): {len(y_gt_filt)}")
p6, r6, f16, _ = precision_recall_fscore_support(
    y_gt_filt, y_pred_filt, labels=RAW_6, average=None, zero_division=0)
p6_w, r6_w, f16_w, _ = precision_recall_fscore_support(
    y_gt_filt, y_pred_filt, labels=RAW_6, average='weighted', zero_division=0)
p6_m, r6_m, f16_m, _ = precision_recall_fscore_support(
    y_gt_filt, y_pred_filt, labels=RAW_6, average='macro', zero_division=0)

print("\nPer-class (Precision / Recall / F1):")
class_names = ["absence","standing","sit-by-bed","sit-on-bed","lying-no-cover","lying-w-cover"]
for raw, p, r, f in zip(RAW_6, p6, r6, f16):
    print(f"  {class_names[RAW_6.index(raw)]:20s}: P={p:.4f}  R={r:.4f}  F1={f:.4f}")
print(f"\n  Weighted avg:  P={p6_w:.4f}  R={r6_w:.4f}  F1={f16_w:.4f}")
print(f"  Macro   avg:   P={p6_m:.4f}  R={r6_m:.4f}  F1={f16_m:.4f}")

print("\nFull classification report (6 classes):")
print(classification_report(y_gt_filt, y_pred_filt, labels=RAW_6,
      target_names=class_names, zero_division=0))

# --- Confusion matrix ---
cm = confusion_matrix(y_gt_filt, y_pred_filt, labels=RAW_6)
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap="Blues")
plt.title("Full Pipeline (5-class CNN) — Confusion Matrix on office1_0")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
cm_path = OUTPUT_DIR / "confusion_matrix.pdf"
plt.savefig(cm_path)
plt.close()
print(f"Confusion matrix saved → {cm_path}")

# --- Save metrics JSON ---
metrics = {
    "overall_accuracy": overall_acc,
    "presence": {"precision": float(p_pres), "recall": float(r_pres), "f1": float(f1_pres)},
    "per_class_6": {
        str(raw): {"precision": float(p), "recall": float(r), "f1": float(f)}
        for raw, p, r, f in zip(RAW_6, p6, r6, f16)
    },
    "weighted_avg": {"precision": float(p6_w), "recall": float(r6_w), "f1": float(f16_w)},
    "macro_avg":    {"precision": float(p6_m), "recall": float(r6_m), "f1": float(f16_m)},
}
metrics_path = OUTPUT_DIR / "metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved → {metrics_path}")

# --- Summary bar chart: per-class F1 ---
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(class_names))
bars = ax.bar(x, f16, color="#7C9D97")
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=30, ha="right")
ax.set_ylim(0, 1)
ax.set_ylabel("F1 Score")
ax.set_title("Full Pipeline (5-class CNN) — Per-Class F1 on office1_0")
for bar, p, r in zip(bars, p6, r6):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"P={p:.2f}\nR={r:.2f}", ha="center", va="bottom", fontsize=7)
plt.tight_layout()
f1_path = OUTPUT_DIR / "per_class_f1.pdf"
plt.savefig(f1_path)
plt.close()
print(f"Per-class F1 chart saved → {f1_path}")

print("\n=== DONE ===")
print(f"  Output dir: {OUTPUT_DIR}")

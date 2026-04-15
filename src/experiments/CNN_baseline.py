"""
Baseline 6-class end-to-end CNN experiment.
Train: office0_4  |  Test: office1_0
Saves: output/case_study/CNN/predictions.json, metrics.json, confusion_matrix.pdf

No data augmentation. Per-frame ThermalInvariantPreprocessor normalization.
Stratified train/val split on the training set.
"""
import sys
import json
from pathlib import Path

WORKSPACE = Path("/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace")
SRC_DIR = WORKSPACE / "src"
sys.path.insert(0, str(SRC_DIR))

OUTPUT_DIR = WORKSPACE / "output" / "case_study" / "CNN"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── imports ─────────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay, precision_recall_fscore_support
)

from dataset.dataset import ThermalDataset
from posture_detection_module.CNN_model import SimpleIRA_CNN
from posture_detection_module.utils import (
    ThermalInvariantPreprocessor, filter_dataset, remap_labels, train_model
)

# ── 6-class label mapping ───────────────────────────────────────────────────
# Labels: 0=absence, 2=standing, 3=sit-by-bed, 4=sit-on-bed, 5=lying-no-cover, 6=lying-w-cover
RAW_LABELS = [0, 2, 3, 4, 5, 6]
NUM_CLASSES = 6
label_to_index = {v: i for i, v in enumerate(RAW_LABELS)}   # {0:0, 2:1, 3:2, 4:3, 5:4, 6:5}
index_to_raw = {i: v for v, i in label_to_index.items()}    # inverse

# ── dataset paths ────────────────────────────────────────────────────────────
TRAIN_PATH = WORKSPACE / "data" / "office0_4"
TEST_PATH = WORKSPACE / "data" / "office1_0"

# ── normalize_and_collate: apply ThermalInvariantPreprocessor per-frame, resize ─
def collate_no_cam(batch):
    """Collate: normalize each frame, resize 62×80 → 60×80, drop 'image'."""
    data_dicts, labels = zip(*batch)
    preproc = ThermalInvariantPreprocessor()
    ira_tensors = []
    for d in data_dicts:
        ira = d["ira_highres"]
        if isinstance(ira, np.ndarray):
            ira = ira.astype(np.float32)
        else:
            ira = ira.cpu().numpy().astype(np.float32)
        # Per-frame normalization (p2-p98 percentile range)
        ira_norm = preproc(ira)  # shape (H, W), values in [0, 1]
        # Resize: (62, 80) → (60, 80)
        ira_t = torch.from_numpy(ira_norm).float().unsqueeze(0).unsqueeze(0)
        ira_t = torch.nn.functional.interpolate(ira_t, size=(60, 80),
                                                mode="bilinear", align_corners=False)
        ira_tensors.append(ira_t.squeeze())   # (60, 80)
    return {"ira_highres": torch.stack(ira_tensors)}, torch.tensor(labels)


# ── load training dataset ──────────────────────────────────────────────────────
print("Loading training dataset (office0_4) …")
train_ds_full = ThermalDataset(str(TRAIN_PATH), noCam=True)
print(f"  Total frames: {len(train_ds_full)}")

# Filter to keep only 6-class labels
train_ds_filt = filter_dataset(train_ds_full, label_to_index)
print(f"  After 6-class filter: {len(train_ds_filt)} frames")

# ── stratified train/val split ───────────────────────────────────────────────
# IMPORTANT: split indices into the FILTERED dataset, not the original
all_filt_indices = list(range(len(train_ds_filt)))
all_filt_labels = [train_ds_filt.dataset.annotations_expanded[train_ds_filt.indices[i]]
                   for i in range(len(train_ds_filt))]

train_idx, val_idx = train_test_split(
    all_filt_indices,
    test_size=0.2,
    stratify=all_filt_labels,
    random_state=42
)
train_subset = Subset(train_ds_filt, train_idx)
val_subset = Subset(train_ds_filt, val_idx)
print(f"  Train: {len(train_subset)}, Val: {len(val_subset)}")

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, collate_fn=collate_no_cam)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, collate_fn=collate_no_cam)

# ── model ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleIRA_CNN(num_classes=NUM_CLASSES).to(device)
print(f"\nModel on: {device}")

# ── training ──────────────────────────────────────────────────────────────────
print("\n=== Training ===")
train_model(model, train_loader, val_loader, label_to_index,
            num_epochs=10, learning_rate=1e-3)

WEIGHT_PATH = WORKSPACE / "weights" / "CNN_baseline_6class.pth"
torch.save(model.state_dict(), WEIGHT_PATH)
print(f"Weights saved → {WEIGHT_PATH}")

# ── load test dataset (FULL office1_0, no filtering) ─────────────────────────
print("\n=== Recording predictions on FULL test set (office1_0) ===")
test_ds_full = ThermalDataset(str(TEST_PATH), noCam=True)
print(f"  Total test frames: {len(test_ds_full)}")

# ── inference on ALL frames ───────────────────────────────────────────────────
model.eval()
preproc = ThermalInvariantPreprocessor()
pred_all = {}   # {frame_idx: predicted_raw_label}

with torch.no_grad():
    for idx in range(len(test_ds_full)):
        # Get raw IRA frame (62, 80)
        ira = test_ds_full.get_ira_highres(idx)
        if isinstance(ira, np.ndarray):
            ira = ira.astype(np.float32)
        else:
            ira = ira.cpu().numpy().astype(np.float32)

        # Per-frame normalization (same as training)
        ira_norm = preproc(ira)

        # Resize (62, 80) → (1, 1, 60, 80) → (1, 60, 80)
        ira_t = torch.from_numpy(ira_norm).float().unsqueeze(0).unsqueeze(0)
        ira_t = torch.nn.functional.interpolate(ira_t, size=(60, 80),
                                                mode="bilinear", align_corners=False)
        ira_t = ira_t.squeeze()   # (60, 80)

        # Forward pass
        out = model(ira_t.unsqueeze(0).to(device))
        pred_idx = torch.argmax(out, dim=1).item()
        pred_all[idx] = index_to_raw[pred_idx]

# Save predictions
pred_path = OUTPUT_DIR / "predictions.json"
with open(pred_path, "w") as f:
    json.dump(pred_all, f, indent=2)
print(f"Predictions saved → {pred_path}")

# ── evaluation: only on frames with valid 6-class GT labels ─────────────────
print("\n=== Evaluation (frames with valid 6-class GT labels) ===")
y_gt = [test_ds_full.annotations_expanded[i] for i in range(len(test_ds_full))]
y_pred = [pred_all[i] for i in range(len(test_ds_full))]

# Keep only frames whose GT label is in our 6-class set
valid_mask = [g in RAW_LABELS for g in y_gt]
y_gt_filt = [y_gt[i] for i in range(len(y_gt)) if valid_mask[i]]
y_pred_filt = [y_pred[i] for i in range(len(y_pred)) if valid_mask[i]]

n = len(y_gt_filt)
print(f"  Frames evaluated (valid 6-class GT): {n}")

overall_acc = sum(p == g for p, g in zip(y_pred_filt, y_gt_filt)) / n
print(f"\nOverall accuracy: {overall_acc:.4f}")

# classification report
print("\nClassification report:")
print(classification_report(y_gt_filt, y_pred_filt, labels=RAW_LABELS,
      target_names=["absence", "standing", "sit-by-bed", "sit-on-bed",
                    "lying-no-cover", "lying-w-cover"]))

# confusion matrix
cm = confusion_matrix(y_gt_filt, y_pred_filt, labels=RAW_LABELS)
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["absence", "standing", "sit-by-bed", "sit-on-bed",
                    "lying-no-cover", "lying-w-cover"]
)
disp.plot(ax=ax, cmap="Blues")
plt.title("Baseline CNN — Confusion Matrix")
plt.tight_layout()
cm_path = OUTPUT_DIR / "confusion_matrix.pdf"
plt.savefig(cm_path)
plt.close()
print(f"Confusion matrix saved → {cm_path}")

# per-class P/R/F1
prec, rec, f1, _ = precision_recall_fscore_support(
    y_gt_filt, y_pred_filt, labels=RAW_LABELS, average=None
)
prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
    y_gt_filt, y_pred_filt, labels=RAW_LABELS, average='weighted'
)
prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
    y_gt_filt, y_pred_filt, labels=RAW_LABELS, average='macro'
)

metrics = {
    "overall_accuracy": overall_acc,
    "per_class": {
        str(raw): {"precision": float(p), "recall": float(r), "f1": float(f)}
        for raw, p, r, f in zip(RAW_LABELS, prec, rec, f1)
    },
    "weighted_avg": {"precision": float(prec_w), "recall": float(rec_w), "f1": float(f1_w)},
    "macro_avg": {"precision": float(prec_m), "recall": float(rec_m), "f1": float(f1_m)}
}
metrics_path = OUTPUT_DIR / "metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved → {metrics_path}")

# ── summary bar chart ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
class_names = ["absence", "standing", "sit-by-bed", "sit-on-bed", "lying-no-cover", "lying-w-cover"]
x = np.arange(len(class_names))
bars = ax.bar(x, f1, color="#7C9D97")
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=30, ha="right")
ax.set_ylim(0, 1)
ax.set_ylabel("F1 Score")
ax.set_title("Baseline CNN — Per-Class F1")
for bar, p, r in zip(bars, prec, rec):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"P={p:.2f}\nR={r:.2f}", ha="center", va="bottom", fontsize=7)
plt.tight_layout()
f1_path = OUTPUT_DIR / "per_class_f1.pdf"
plt.savefig(f1_path)
plt.close()
print(f"Per-class F1 chart saved → {f1_path}")

print("\n=== DONE ===")
print(f"  weights/  : {WEIGHT_PATH}")
print(f"  output dir: {OUTPUT_DIR}")
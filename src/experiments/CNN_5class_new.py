"""
5-class posture CNN experiment (same architecture as CNN_baseline).
Train: office0_0, office0_2, office0_3, office0_4  |  Test: office1_0
Saves: output/case_study/CNN_5class_new/predictions.json, metrics.json, confusion_matrix.pdf
"""
import sys
import json
from pathlib import Path

WORKSPACE = Path("/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace")
SRC_DIR = WORKSPACE / "src"
sys.path.insert(0, str(SRC_DIR))

OUTPUT_DIR = WORKSPACE / "output" / "case_study" / "CNN_5class_new"
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
from posture_detection_module.utils import ThermalInvariantPreprocessor, remap_labels
from torch.utils.data import Dataset

# ── 5-class label mapping (postures only, no absence) ────────────────────
# Labels: 2=standing, 3=sit-by-bed, 4=sit-on-bed, 5=lying-no-cover, 6=lying-w-cover
RAW_LABELS = [2, 3, 4, 5, 6]
NUM_CLASSES = 5
label_to_index = {v: i for i, v in enumerate(RAW_LABELS)}   # {2:0, 3:1, 4:2, 5:3, 6:4}
index_to_raw = {i: v for v, i in label_to_index.items()}    # inverse

# ── dataset paths ────────────────────────────────────────────────────────────
TRAIN_PATHS = [
    WORKSPACE / "data" / "office0_0",
    WORKSPACE / "data" / "office0_2",
    WORKSPACE / "data" / "office0_3",
    WORKSPACE / "data" / "office0_4",
]
TEST_PATH = WORKSPACE / "data" / "office1_0"

# ── Simple list-based Dataset for multi-source training ──────────────────────
class FlatMultiDataset(Dataset):
    """Flatten multiple ThermalDatasets into a single dataset with (ds_idx, frame_idx)."""
    def __init__(self, datasets, entries):
        self.datasets = datasets
        self.entries = entries  # list of (ds_idx, frame_idx, label)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        ds_idx, frame_idx, label = self.entries[idx]
        ira = self.datasets[ds_idx].get_ira_highres(frame_idx)
        return {"ira_highres": ira}, label


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


# ── load training datasets ───────────────────────────────────────────────────
print("Loading training datasets …")
all_datasets = []
all_entries = []  # (ds_idx, frame_idx, label)

for ds_idx, path in enumerate(TRAIN_PATHS):
    ds = ThermalDataset(str(path), noCam=True)
    all_datasets.append(ds)
    for frame_idx, label in enumerate(ds.annotations_expanded):
        if label in label_to_index:
            all_entries.append((ds_idx, frame_idx, label))
    print(f"  {path.name}: {len(ds)} frames, {sum(1 for e in all_entries if e[0]==ds_idx)} valid")

print(f"  Total after 5-class filter: {len(all_entries)} frames")

# Create flat dataset
flat_ds = FlatMultiDataset(all_datasets, all_entries)

# ── stratified train/val split ───────────────────────────────────────────────
all_labels = [e[2] for e in all_entries]
all_indices = list(range(len(all_entries)))
train_idx, val_idx = train_test_split(
    all_indices,
    test_size=0.2,
    stratify=all_labels,
    random_state=42
)
train_subset = Subset(flat_ds, train_idx)
val_subset = Subset(flat_ds, val_idx)
print(f"  Train: {len(train_subset)}, Val: {len(val_subset)}")

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, collate_fn=collate_no_cam)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, collate_fn=collate_no_cam)

# ── model ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleIRA_CNN(num_classes=NUM_CLASSES).to(device)
print(f"\nModel on: {device}")

# ── training ──────────────────────────────────────────────────────────────────
print("\n=== Training ===")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        thermal = batch[0]["ira_highres"].to(device)
        labels = remap_labels(batch[1], label_to_index, device)
        out = model(thermal)
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    # Validate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            thermal = batch[0]["ira_highres"].to(device)
            labels = remap_labels(batch[1], label_to_index, device)
            out = model(thermal)
            pred = out.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total
    print(f"Epoch {epoch+1}/{num_epochs} — Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

WEIGHT_PATH = WORKSPACE / "weights" / "CNN_5class_new.pth"
torch.save(model.state_dict(), WEIGHT_PATH)
print(f"Weights saved → {WEIGHT_PATH}")

# ── load test dataset (FULL office1_0, no filtering) ─────────────────────────
print("\n=== Recording predictions on FULL test set (office1_0) ===")
test_ds_full = ThermalDataset(str(TEST_PATH), noCam=True)
print(f"  Total test frames: {len(test_ds_full)}")

# ── inference on ALL frames ───────────────────────────────────────────────
model.eval()
preproc = ThermalInvariantPreprocessor()
pred_all = {}   # {frame_idx: predicted_raw_label}

with torch.no_grad():
    for idx in range(len(test_ds_full)):
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
        ira_t = ira_t.squeeze()

        # Forward pass
        out = model(ira_t.unsqueeze(0).to(device))
        pred_idx = torch.argmax(out, dim=1).item()
        pred_all[idx] = index_to_raw[pred_idx]

# Save predictions as array (for confusion.py compatibility)
pred_path = OUTPUT_DIR / "predictions.json"
pred_list = [pred_all[i] for i in range(len(pred_all))]
with open(pred_path, "w") as f:
    json.dump(pred_list, f, indent=2)
print(f"Predictions saved → {pred_path}")

# ── evaluation: only on frames with valid 5-class GT labels ─────────────────
print("\n=== Evaluation (frames with valid 5-class GT labels) ===")
y_gt = [test_ds_full.annotations_expanded[i] for i in range(len(test_ds_full))]
y_pred = [pred_all[i] for i in range(len(test_ds_full))]

# Keep only frames whose GT label is in our 5-class set
valid_mask = [g in RAW_LABELS for g in y_gt]
y_gt_filt = [y_gt[i] for i in range(len(y_gt)) if valid_mask[i]]
y_pred_filt = [y_pred[i] for i in range(len(y_pred)) if valid_mask[i]]

n = len(y_gt_filt)
print(f"  Frames evaluated (valid 5-class GT): {n}")

overall_acc = sum(p == g for p, g in zip(y_pred_filt, y_gt_filt)) / n
print(f"\nOverall accuracy: {overall_acc:.4f}")

# classification report
print("\nClassification report:")
print(classification_report(y_gt_filt, y_pred_filt, labels=RAW_LABELS,
      target_names=["standing", "sit-by-bed", "sit-on-bed", "lying-no-cover", "lying-w-cover"]))

# confusion matrix
cm = confusion_matrix(y_gt_filt, y_pred_filt, labels=RAW_LABELS)
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["standing", "sit-by-bed", "sit-on-bed", "lying-no-cover", "lying-w-cover"]
)
disp.plot(ax=ax, cmap="Blues")
plt.title("CNN_5class_new — Confusion Matrix")
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
class_names = ["standing", "sit-by-bed", "sit-on-bed", "lying-no-cover", "lying-w-cover"]
x = np.arange(len(class_names))
bars = ax.bar(x, f1, color="#7C9D97")
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=30, ha="right")
ax.set_ylim(0, 1)
ax.set_ylabel("F1 Score")
ax.set_title("CNN_5class_new — Per-Class F1")
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

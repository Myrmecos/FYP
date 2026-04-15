"""
LSTM Baseline for temporal bed posture classification.
Train: office0_0, office0_2, office0_3, office0_4  |  Test: office1_0
Saves: output/LSTM/predictions.json, metrics.json, confusion_matrix.pdf
"""
import sys, os
from pathlib import Path
import random
import numpy as np
import torch

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── path setup ──────────────────────────────────────────────────────────────
WORKSPACE = Path("/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace")
SRC_DIR   = WORKSPACE / "src"
sys.path.insert(0, str(SRC_DIR))

OUTPUT_DIR = WORKSPACE / "output" / "case_study" / "LSTM"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── imports ─────────────────────────────────────────────────────────────────
import json
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
from tqdm import tqdm

from dataset.dataset import ThermalDataset, ThermalDatasetAggregator
from posture_detection_module.utils import (
    ThermalInvariantPreprocessor, filter_dataset, remap_labels
)

# ── 6-class label mapping ───────────────────────────────────────────────────
RAW_LABELS  = [0, 2, 3, 4, 5, 6]          # 6 classes: absence + 5 postures
NUM_CLASSES = 6
label_to_index = {v: i for i, v in enumerate(RAW_LABELS)}
index_to_raw   = {i: v for v, i in label_to_index.items()}

# ── LSTM Model ──────────────────────────────────────────────────────────────
class SimpleIRA_LSTM(nn.Module):
    """CNN feature extractor + LSTM for temporal modeling."""
    def __init__(self, num_classes=6, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()

        # CNN backbone (same as SimpleIRA_CNN)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # After conv + pooling: 128 * 7 * 10 = 8960
        self.fc_cnn = nn.Linear(128 * 7 * 10, 512)

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Classification head
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, seq_len, H, W) = (B, seq_len, 60, 80)
        B, seq_len, H, W = x.shape

        # Reshape for CNN: merge batch and sequence
        x = x.view(B * seq_len, 1, H, W)  # (B*seq, 1, 60, 80)

        # CNN feature extraction
        x = self.relu(self.conv1(x))
        x = self.pool(x)                     # (B*seq, 32, 30, 40)

        x = self.relu(self.conv2(x))
        x = self.pool(x)                     # (B*seq, 64, 15, 20)

        x = self.relu(self.conv3(x))
        x = self.pool(x)                     # (B*seq, 128, 7, 10)

        # Flatten and project
        x = x.view(B * seq_len, -1)          # (B*seq, 8960)
        x = self.relu(self.fc_cnn(x))        # (B*seq, 512)
        x = self.dropout(x)

        # Reshape for LSTM: (B, seq, 512)
        x = x.view(B, seq_len, 512)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (B, seq, hidden)

        # Use last timestep output
        x = lstm_out[:, -1, :]               # (B, hidden_size)

        # Classification head
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                       # (B, num_classes)

        return x


# ── Sequence Dataset ────────────────────────────────────────────────────────
class SequenceDataset(torch.utils.data.Dataset):
    """Creates sequences of thermal frames from a ThermalDataset, Subset, or ThermalDatasetAggregator."""

    def __init__(self, dataset, seq_len=16, stride=8):
        self.dataset = dataset
        self.seq_len = seq_len
        self.stride = stride
        self.preprocessor = ThermalInvariantPreprocessor()

        # Handle ThermalDatasetAggregator
        if hasattr(dataset, 'datasets') and hasattr(dataset, 'annotations'):
            # It's a ThermalDatasetAggregator - use its combined annotations
            self.annotations = dataset.annotations
            self.frame_indices = list(range(len(dataset)))
            self.get_ira_func = dataset.get_ira_highres
        # Handle Subset (from filter_dataset)
        elif hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
            # It's a Subset - access underlying dataset and indices
            self.annotations = dataset.dataset.annotations_expanded
            self.frame_indices = dataset.indices  # list of valid frame indices
            self.get_ira_func = dataset.dataset.get_ira_highres
        else:
            # It's a regular ThermalDataset
            self.annotations = dataset.annotations_expanded
            self.frame_indices = list(range(len(dataset)))
            self.get_ira_func = dataset.get_ira_highres

        total_frames = len(self.frame_indices)

        # Build valid sequence indices (sequences where all frames have valid labels)
        self.valid_seq_indices = []  # stores index into self.frame_indices

        for start_pos in range(0, total_frames - seq_len + 1, stride):
            end_pos = start_pos + seq_len
            frame_idx_range = self.frame_indices[start_pos:end_pos]
            labels = [self.annotations[i] for i in frame_idx_range]
            # Only include sequences where all frames have valid labels
            if all(label in label_to_index for label in labels):
                self.valid_seq_indices.append(start_pos)

    def __len__(self):
        return len(self.valid_seq_indices)

    def __getitem__(self, idx):
        start_pos = self.valid_seq_indices[idx]
        seq_len = self.seq_len

        # Collect sequence of thermal frames
        seq_tensors = []
        for pos in range(start_pos, start_pos + seq_len):
            frame_idx = self.frame_indices[pos]
            ira = self.get_ira_func(frame_idx)
            if isinstance(ira, np.ndarray):
                ira_norm = self.preprocessor(ira)
            else:
                ira_norm = self.preprocessor(ira.cpu().numpy())
            seq_tensors.append(torch.from_numpy(ira_norm).float())

        # Stack: (seq_len, H, W) = (16, 60, 80)
        seq_tensor = torch.stack(seq_tensors)

        # Label: use the last frame's label
        last_frame_idx = self.frame_indices[start_pos + seq_len - 1]
        label = self.annotations[last_frame_idx]

        return seq_tensor, label


# ── dataset paths ────────────────────────────────────────────────────────────
TRAIN_PATHS = [
    WORKSPACE / "data" / "office0_0",
    WORKSPACE / "data" / "office0_2",
    WORKSPACE / "data" / "office0_3",
    WORKSPACE / "data" / "office0_4",
]
TEST_PATH  = WORKSPACE / "data" / "office1_0"

# ── load datasets ────────────────────────────────────────────────────────────
print("Loading training datasets …")
train_datasets = []
for path in TRAIN_PATHS:
    ds = ThermalDataset(str(path), noCam=True)
    ds = filter_dataset(ds, label_to_index)
    train_datasets.append(ds)
    print(f"  {path.name}: {len(ds)} frames")

# Use ThermalDatasetAggregator to combine training sets
train_ds = ThermalDatasetAggregator([str(p) for p in TRAIN_PATHS])
print(f"  Combined training samples (frames) after filter: {len(train_ds)}")

print("Loading test dataset …")
test_ds = ThermalDataset(str(TEST_PATH), noCam=True)
test_ds = filter_dataset(test_ds, label_to_index)
print(f"  Test samples (frames) after filter: {len(test_ds)}")

# ── create sequence datasets ────────────────────────────────────────────────────
SEQ_LEN = 16
STRIDE = 8

print(f"\nCreating sequences (seq_len={SEQ_LEN}, stride={STRIDE}) …")
train_seq_ds = SequenceDataset(train_ds, seq_len=SEQ_LEN, stride=STRIDE)
test_seq_ds = SequenceDataset(test_ds, seq_len=SEQ_LEN, stride=STRIDE)
print(f"  Training sequences: {len(train_seq_ds)}")
print(f"  Test sequences: {len(test_seq_ds)}")

# ── stratified train/val split ────────────────────────────────────────────────
train_labels = [train_seq_ds[i][1] for i in range(len(train_seq_ds))]
train_indices = list(range(len(train_seq_ds)))
train_indices, val_indices = train_test_split(
    train_indices, test_size=0.2, stratify=train_labels, random_state=42
)
train_subset = Subset(train_seq_ds, train_indices)
val_subset   = Subset(train_seq_ds, val_indices)
print(f"  Train: {len(train_subset)}, Val: {len(val_subset)}")


def collate_seq(batch):
    """Collate a batch of sequences."""
    seqs, labels = zip(*batch)
    seqs_stacked = torch.stack(seqs)           # (N, seq_len, H, W)
    labels_tensor = torch.tensor(labels)
    return {"ira_highres": seqs_stacked}, labels_tensor


train_loader = DataLoader(train_subset, batch_size=16, shuffle=True, collate_fn=collate_seq)
val_loader   = DataLoader(val_subset,   batch_size=16, shuffle=False, collate_fn=collate_seq)

# ── model ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleIRA_LSTM(num_classes=NUM_CLASSES).to(device)
print(f"\nModel on: {device}")
print(model)


# ── training ──────────────────────────────────────────────────────────────────
print("\n=== Training ===")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        thermal = batch[0]['ira_highres'].to(device=device, dtype=torch.float32)
        labels = remap_labels(batch[1], label_to_index, device)

        outputs = model(thermal)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Validate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            thermal = batch[0]['ira_highres'].to(device=device, dtype=torch.float32)
            labels = remap_labels(batch[1], label_to_index, device)

            outputs = model(thermal)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total
    print(f"Validation Accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        WEIGHT_PATH = WORKSPACE / "weights" / "LSTM_baseline_6class.pth"
        torch.save(model.state_dict(), WEIGHT_PATH)
        print(f"  -> Best model saved (acc={val_acc:.4f})")

print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")

# ── load best model ──────────────────────────────────────────────────────────
WEIGHT_PATH = WORKSPACE / "weights" / "LSTM_baseline_6class.pth"
model.load_state_dict(torch.load(WEIGHT_PATH))
print(f"Loaded best model from {WEIGHT_PATH}")

# ── record predictions on FULL test set ──────────────────────────────────────
print("\n=== Recording predictions on test set ===")
model.eval()
pred_records = {}

test_loader_full = DataLoader(test_seq_ds, batch_size=16, shuffle=False, collate_fn=collate_seq)
with torch.no_grad():
    for batch in tqdm(test_loader_full, desc="Predicting"):
        thermal = batch[0]['ira_highres'].to(device=device, dtype=torch.float32)
        labels_raw = batch[1].numpy()

        outputs = model(thermal)
        pred_indices = torch.argmax(outputs, dim=1).cpu().numpy()

        for i, (pred_idx, label_raw) in enumerate(zip(pred_indices, labels_raw)):
            seq_start = i * 16  # approximate mapping (stride=8 means overlap)
            pred_raw = index_to_raw[pred_idx]
            pred_records[len(pred_records)] = int(pred_raw)

# save predictions
pred_path = OUTPUT_DIR / "predictions.json"
with open(pred_path, "w") as f:
    json.dump(pred_records, f, indent=2)
print(f"Predictions saved → {pred_path}")

# ── evaluation ────────────────────────────────────────────────────────────────
print("\n=== Evaluation ===")
gt_records = {i: test_seq_ds[i][1] for i in range(len(test_seq_ds))}

# align predictions & ground truth
y_gt   = [gt_records[i]   for i in range(len(test_seq_ds))]
y_pred = [pred_records[i]  for i in range(len(test_seq_ds))]

overall_acc = sum(p == g for p, g in zip(y_pred, y_gt)) / len(y_gt)
print(f"\nOverall accuracy: {overall_acc:.4f}")

# classification report
print("\nClassification report:")
print(classification_report(y_gt, y_pred, labels=RAW_LABELS,
      target_names=["absence","standing","sit-by-bed","sit-on-bed","lying-no-cover","lying-w-cover"]))

# confusion matrix
cm = confusion_matrix(y_gt, y_pred, labels=RAW_LABELS)
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
      display_labels=["absence","standing","sit-by-bed","sit-on-bed","lying-no-cover","lying-w-cover"])
disp.plot(ax=ax, cmap="Blues")
plt.title("Baseline LSTM — Confusion Matrix")
plt.tight_layout()
cm_path = OUTPUT_DIR / "confusion_matrix.pdf"
plt.savefig(cm_path)
plt.close()
print(f"Confusion matrix saved → {cm_path}")

# per-class P/R/F1
prec, rec, f1, _ = precision_recall_fscore_support(y_gt, y_pred, labels=RAW_LABELS, average=None)
metrics = {
    "overall_accuracy": overall_acc,
    "per_class": {
        raw: {"precision": float(p), "recall": float(r), "f1": float(f)}
        for raw, p, r, f in zip(RAW_LABELS, prec, rec, f1)
    }
}
metrics_path = OUTPUT_DIR / "metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved → {metrics_path}")

# ── summary bar chart ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
class_names = ["absence","standing","sit-by-bed","sit-on-bed","lying-no-cover","lying-w-cover"]
x = np.arange(len(class_names))
bars = ax.bar(x, f1, color="#5B9BD5")
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=30, ha="right")
ax.set_ylim(0, 1)
ax.set_ylabel("F1 Score")
ax.set_title("Baseline LSTM — Per-Class F1")
for bar, p, r in zip(bars, prec, rec):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"P={p:.2f}\nR={r:.2f}", ha="center", va="bottom", fontsize=7)
plt.tight_layout()
f1_path = OUTPUT_DIR / "per_class_f1.pdf"
plt.savefig(f1_path)
plt.close()
print(f"Per-class F1 chart saved → {f1_path}")

print("\n=== DONE ===")
print(f"  weights/  : {WEIGHT_PATH}")
print(f"  output dir: {OUTPUT_DIR}")

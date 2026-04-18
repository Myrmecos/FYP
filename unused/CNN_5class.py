"""
Train 5-class posture CNN (same architecture as existing posture detector).
Train: office0_0, office0_2, office0_3, office0_4  |  Test: office1_0
Saves: weights/CNN_5class_posture.pth
"""
import sys, os
from pathlib import Path

WORKSPACE = Path("/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace")
SRC_DIR   = WORKSPACE / "src"
sys.path.insert(0, str(SRC_DIR))

# ── imports ─────────────────────────────────────────────────────────────────
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset.dataset import ThermalDataset
from posture_detection_module.CNN_model import SimpleIRA_CNN
from posture_detection_module.utils import ThermalNormalize, filter_dataset, remap_labels

# ── 5-class label mapping (postures only, no absence) ───────────────────────
RAW_LABELS_5  = [2, 3, 4, 5, 6]   # standing, sit-by-bed, sit-on-bed, lying-no-cover, lying-w-cover
NUM_CLASSES    = 5
label_to_index = {v: i for i, v in enumerate(RAW_LABELS_5)}   # {2:0, 3:1, 4:2, 5:3, 6:4}
index_to_raw   = {i: v for v, i in label_to_index.items()}    # inverse

# ── dataset paths ────────────────────────────────────────────────────────────
TRAIN_PATHS = [WORKSPACE / "data" / f"office0_{n}" for n in [4]]

# ── load all datasets, collect valid (ds_idx, frame_idx, label) tuples ─────
print("Loading training datasets …")
all_datasets = []
frame_entries = []   # (ds_idx, frame_idx, label)
for ds_idx, p in enumerate(TRAIN_PATHS):
    ds = ThermalDataset(str(p), noCam=True)
    all_datasets.append(ds)
    for frame_idx, label in enumerate(ds.annotations_expanded):
        if label in label_to_index:
            frame_entries.append((ds_idx, frame_idx, label))
print(f"  Total combined training samples after filter: {len(frame_entries)}")

# ── simple list-based Dataset wrapper ───────────────────────────────────────
class FlatDataset(Dataset):
    def __init__(self, datasets, entries):
        self.datasets = datasets
        self.entries = entries
    def __len__(self):
        return len(self.entries)
    def __getitem__(self, idx):
        ds_idx, frame_idx, label = self.entries[idx]
        ira_highres = self.datasets[ds_idx].get_ira_highres(frame_idx)
        return {"ira_highres": ira_highres}, label

flat_ds = FlatDataset(all_datasets, frame_entries)

# ── stratified train/val split ───────────────────────────────────────────────
all_labels = [e[2] for e in frame_entries]
all_indices = list(range(len(frame_entries)))
train_indices, val_indices = train_test_split(
    all_indices, test_size=0.2, stratify=all_labels, random_state=42
)
train_subset = Subset(flat_ds, train_indices)
val_subset   = Subset(flat_ds, val_indices)
print(f"  Train: {len(train_subset)}, Val: {len(val_subset)}")


# ── collate_fn (resize 62×80 → 60×80) ─────────────────────────────────────────
def collate_5class(batch):
    data_dicts, labels = zip(*batch)
    ira_tensors = []
    for d in data_dicts:
        ira = d["ira_highres"]
        if isinstance(ira, np.ndarray):
            ira = torch.from_numpy(ira)
        # (62, 80) → (1, 1, 62, 80) → (1, 1, 60, 80) → (60, 80)
        ira = ira.float().unsqueeze(0).unsqueeze(0)
        ira = torch.nn.functional.interpolate(ira, size=(60, 80),
                                              mode="bilinear", align_corners=False)
        ira_tensors.append(ira.squeeze(0).squeeze(0))
    collated = {"ira_highres": torch.stack(ira_tensors)}
    return collated, torch.tensor(labels)

train_loader = DataLoader(
    ThermalNormalize(train_subset, augment=False),
    batch_size=32, shuffle=True, collate_fn=collate_5class
)
val_loader = DataLoader(
    ThermalNormalize(val_subset, augment=False),
    batch_size=32, shuffle=False, collate_fn=collate_5class
)

# ── model & training setup ───────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = SimpleIRA_CNN(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ── training loop ────────────────────────────────────────────────────────────
NUM_EPOCHS = 10
best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    # Train
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        thermal = batch[0]["ira_highres"].to(device)   # (N, 60, 80)
        labels  = remap_labels(batch[1], label_to_index, device)
        out     = model(thermal)
        loss    = criterion(out, labels)
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
            labels  = remap_labels(batch[1], label_to_index, device)
            out     = model(thermal)
            pred    = torch.argmax(out, dim=1)
            correct += (pred == labels).sum().item()
            total   += labels.size(0)
    val_acc = correct / total
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} — Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc

# ── save weights ─────────────────────────────────────────────────────────────
WEIGHT_PATH = WORKSPACE / "weights" / "CNN_5class_posture.pth"
torch.save(model.state_dict(), WEIGHT_PATH)
print(f"\nWeights saved → {WEIGHT_PATH} (best val acc: {best_val_acc:.4f})")

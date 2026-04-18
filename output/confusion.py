"""
Confusion Matrix Visualization Script
Reads predictions.json and generates a confusion matrix with rotated labels.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ── Configuration ───────────────────────────────────────────────────────────
PREDICTIONS_PATH = "output/case_study/pipeline_5class/predictions.json"
OUTPUT_PATH = "output/case_study/pipeline_5class/confusion_matrix_rotated.pdf"

# 6-class label mapping
RAW_LABELS = [0, 2, 3, 4, 5, 6]
LABEL_NAMES = ["absence", "standing", "sit-by-bed", "sit-on-bed", "lying-no-cover", "lying-w-cover"]

# ── Load ground truth ───────────────────────────────────────────────────────
# Since predictions.json only contains predictions, we need to reconstruct
# ground truth from the test dataset annotations
# The indices in predictions.json correspond to test sequence/frame indices

import sys
from pathlib import Path
WORKSPACE = Path("/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace")
sys.path.insert(0, str(WORKSPACE / "src"))

from dataset.dataset import ThermalDataset, ThermalDatasetAggregator
from posture_detection_module.utils import filter_dataset, ThermalInvariantPreprocessor

# Rebuild test dataset (same as LSTM_baseline.py)
TEST_PATH = WORKSPACE / "data" / "office1_0"
test_ds = ThermalDataset(str(TEST_PATH), noCam=True)
label_to_index = {v: i for i, v in enumerate(RAW_LABELS)}
test_ds_filtered = filter_dataset(test_ds, label_to_index)

# For CNN-style: get ALL ground truth labels (including invalid ones as -1)
# to match CNN's per-frame predictions which iterates through all frames
all_gt_labels = test_ds.annotations_expanded

# For LSTM: predictions are per-sequence, need to match with sequence ground truth
# Rebuild sequence dataset to get correct gt alignment
SEQ_LEN = 16
STRIDE = 8

# Check if this is LSTM predictions (shorter) or CNN predictions (longer)
with open(PREDICTIONS_PATH, 'r') as f:
    pred_data = json.load(f)

# Handle both JSON object {"0": val, "1": val, ...} and JSON array [val, val, ...]
if isinstance(pred_data, dict):
    num_preds = len(pred_data)
    pred_data_is_dict = True
else:
    num_preds = len(pred_data)
    pred_data_is_dict = False

if num_preds < 1000:
    # LSTM-style: predictions are per-sequence
    print(f"Detected LSTM-style predictions ({num_preds} sequences)")

    # Rebuild valid sequence indices (same logic as SequenceDataset)
    preprocessor = ThermalInvariantPreprocessor()
    annotations = test_ds_filtered.dataset.annotations_expanded
    frame_indices = test_ds_filtered.indices
    valid_seq_indices = []

    total_frames = len(frame_indices)
    for start_pos in range(0, total_frames - SEQ_LEN + 1, STRIDE):
        end_pos = start_pos + SEQ_LEN
        frame_idx_range = frame_indices[start_pos:end_pos]
        labels = [annotations[i] for i in frame_idx_range]
        if all(label in label_to_index for label in labels):
            valid_seq_indices.append(start_pos)

    # Ground truth for sequences (last frame's label in each sequence)
    y_true = [annotations[frame_indices[pos + SEQ_LEN - 1]] for pos in valid_seq_indices]
    if pred_data_is_dict:
        y_pred = [int(pred_data[str(i)]) for i in range(len(valid_seq_indices))]
    else:
        y_pred = [int(pred_data[i]) for i in range(len(valid_seq_indices))]
else:
    # CNN-style: predictions are per-frame
    print(f"Detected CNN-style predictions ({num_preds} frames)")
    y_true = all_gt_labels[:num_preds]
    if pred_data_is_dict:
        y_pred = [int(pred_data[str(i)]) for i in range(num_preds)]
    else:
        y_pred = [int(pred_data[i]) for i in range(num_preds)]

print(f"Total predictions: {len(y_pred)}")
print(f"Ground truth samples: {len(y_true)}")

# ── Compute and plot confusion matrix ───────────────────────────────────────
cm = confusion_matrix(y_true, y_pred, labels=RAW_LABELS)

fig, ax = plt.subplots(figsize=(12, 10))

# Use ConfusionMatrixDisplay - disable internal values first
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
disp.plot(ax=ax, cmap="Blues", include_values=False)  # disable default values

# Add text annotations manually with controlled font size
# Get the display values (normalized or raw counts)
cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalize by row (true label)
cm_flat = cm.flatten()
cm_display_flat = cm_display.flatten()

for i in range(len(cm_flat)):
    row, col = i // cm.shape[1], i % cm.shape[1]
    val = cm_display_flat[i]
    txt_color = 'white' if val > 0.5 else 'black'
    ax.text(col, row, f'{cm_flat[i]:.0f}', ha='center', va='center',
            fontsize=18, color=txt_color, fontweight='bold')

# Rotate x-axis labels by 45 degrees
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=19)
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=18)

# Increase label sizes
ax.set_xlabel('Predicted Label', fontsize=18)
ax.set_ylabel('True Label', fontsize=18)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save figure
plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=150)
print(f"Confusion matrix saved → {OUTPUT_PATH}")

# Also save as PNG for easy viewing
png_path = OUTPUT_PATH.replace(".pdf", ".png")
plt.savefig(png_path, bbox_inches="tight", dpi=150)
print(f"PNG saved → {png_path}")

plt.close()

# ── Print summary statistics ──────────────────────────────────────────────────
valid_mask = [g in RAW_LABELS for g in y_true]
y_true = [y_true[i] for i in range(len(y_true)) if valid_mask[i]]
y_pred = [y_pred[i] for i in range(len(y_pred)) if valid_mask[i]]

from sklearn.metrics import classification_report, accuracy_score

print(f"\nOverall Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print("\nClassification Report:")
# print 4 decimals in classification report
print(classification_report(y_true, y_pred, labels=RAW_LABELS, target_names=LABEL_NAMES, digits = 4))

# print confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=RAW_LABELS)
print("\nConfusion Matrix (rows=true labels, cols=predicted labels):")
print("Labels:", LABEL_NAMES)
print(cm)

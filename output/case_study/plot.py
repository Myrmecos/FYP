"""
Utility plotting script for case_study results.
Reads predictions.json (either dict {"0": val, ...} or list [val, ...])
and aligns them with ground-truth labels from the test dataset.

Generates:
  - GT vs. Predicted time-series
  - Confusion matrix
  - Per-class metrics bar chart
"""
import json, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import (
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay, accuracy_score,
    precision_recall_fscore_support
)

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKSPACE = Path("/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace")
PREDICTIONS_PATH = WORKSPACE / "output/case_study/CNN/predictions.json"
TEST_PATH = WORKSPACE / "data" / "office1_0"

sys.path.insert(0, str(WORKSPACE / "src"))
from dataset.dataset import ThermalDataset
from posture_detection_module.utils import filter_dataset

# ── Label definitions ──────────────────────────────────────────────────────────
RAW_LABELS = [0, 2, 3, 4, 5, 6]
LABEL_NAMES = ["absence", "standing", "sit-by-bed", "sit-on-bed", "lying-no-cover", "lying-w-cover"]
NUM_CLASSES = len(RAW_LABELS)

# ── Load predictions ────────────────────────────────────────────────────────────
with open(PREDICTIONS_PATH, "r") as f:
    raw_pred = json.load(f)

if isinstance(raw_pred, dict):
    num_preds = len(raw_pred)
    y_pred = [int(raw_pred[str(i)]) for i in range(num_preds)]
else:
    num_preds = len(raw_pred)
    y_pred = [int(v) for v in raw_pred]

print(f"Loaded {num_preds} predictions from {PREDICTIONS_PATH.name}")

# ── Load ground truth ───────────────────────────────────────────────────────────
test_ds = ThermalDataset(str(TEST_PATH), noCam=True)
all_gt = test_ds.annotations_expanded[:num_preds]   # align length

print(f"Loaded {len(all_gt)} ground-truth labels from {TEST_PATH.name}")
print(f"GT distribution: {dict(zip(*np.unique(all_gt, return_counts=True)))}")

# ── Metrics ────────────────────────────────────────────────────────────────────
overall_acc = accuracy_score(all_gt, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(
    all_gt, y_pred, labels=RAW_LABELS, average=None, zero_division=0
)
prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
    all_gt, y_pred, labels=RAW_LABELS, average="weighted", zero_division=0
)
prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
    all_gt, y_pred, labels=RAW_LABELS, average="macro", zero_division=0
)

print(f"\nOverall Accuracy: {overall_acc:.4f}")
print(f"Weighted  P={prec_w:.3f} R={rec_w:.3f} F1={f1_w:.3f}")
print(f"Macro      P={prec_m:.3f} R={rec_m:.3f} F1={f1_m:.3f}")
print("\nClassification Report:")
print(classification_report(all_gt, y_pred, labels=RAW_LABELS,
      target_names=LABEL_NAMES, zero_division=0))

# ── Confusion Matrix ───────────────────────────────────────────────────────────
cm = confusion_matrix(all_gt, y_pred, labels=RAW_LABELS)

fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
disp.plot(ax=ax_cm, cmap="Blues", include_values=False)

cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
for i, row in enumerate(cm):
    for j, val in enumerate(row):
        frac = cm_norm[i, j]
        color = "white" if frac > 0.5 else "black"
        ax_cm.text(j, i, f"{val}\n({frac:.0%})", ha="center", va="center",
                   fontsize=11, color=color, fontweight="bold")

ax_cm.set_xlabel("Predicted Label", fontsize=13)
ax_cm.set_ylabel("True Label", fontsize=13)
plt.setp(ax_cm.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
cm_path = PREDICTIONS_PATH.parent / "gt_pred_confusion_matrix.pdf"
fig_cm.savefig(cm_path, dpi=150)
print(f"Confusion matrix saved → {cm_path}")
plt.close(fig_cm)

# ── GT vs. Predicted Time Series ───────────────────────────────────────────────
# downsample for visibility
DOWN = 1  # plot every frame; set > 1 to downsample
frames = np.arange(0, num_preds, DOWN)
gt_arr = np.array(all_gt)[::DOWN]
pr_arr = np.array(y_pred)[::DOWN]

# collapse to binary (presence=1: labels 2-6, absence=0)
gt_binary = (np.array(gt_arr) != 0).astype(int)
pr_binary = (np.array(pr_arr) != 0).astype(int)

fig_ts, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)

# Panel 1: full 6-class GT
colors_6 = {0: "lightgray", 2: "tab:blue", 3: "tab:orange",
            4: "tab:green", 5: "tab:red", 6: "tab:purple"}
y_colors_gt = [colors_6.get(v, "lightgray") for v in gt_arr]
axes[0].scatter(frames, gt_arr, c=y_colors_gt, s=2, marker="|")
axes[0].set_ylabel("GT Label", fontsize=12)
axes[0].set_yticks(RAW_LABELS)
axes[0].set_yticklabels(LABEL_NAMES)
axes[0].set_ylim(-0.5, 6.5)
axes[0].set_title(f"Ground Truth — {TEST_PATH.name}", fontsize=13)
axes[0].grid(True, alpha=0.3)

# Panel 2: full 6-class Pred
y_colors_pr = [colors_6.get(v, "lightgray") for v in pr_arr]
axes[1].scatter(frames, pr_arr, c=y_colors_pr, s=2, marker="|")
axes[1].set_ylabel("Pred Label", fontsize=12)
axes[1].set_yticks(RAW_LABELS)
axes[1].set_yticklabels(LABEL_NAMES)
axes[1].set_ylim(-0.5, 6.5)
axes[1].set_title(f"Predictions — {PREDICTIONS_PATH.name}", fontsize=13)
axes[1].grid(True, alpha=0.3)

# Panel 3: binary presence/absence
match = (gt_binary == pr_binary)
axes[2].scatter(frames[match], pr_binary[match], c="tab:blue", s=3, marker="|", label="correct")
axes[2].scatter(frames[~match], pr_binary[~match], c="tab:red", s=3, marker="|", label="wrong")
axes[2].set_ylabel("Pred Presence", fontsize=12)
axes[2].set_xlabel("Frame Index", fontsize=12)
axes[2].set_yticks([0, 1])
axes[2].set_yticklabels(["absence", "present"])
axes[2].set_ylim(-0.2, 1.2)
axes[2].set_title(f"Binary Presence (accuracy={overall_acc:.3f})", fontsize=13)
axes[2].legend(loc="upper right")
axes[2].grid(True, alpha=0.3)

# legend patches
legend_patches = [Patch(facecolor=c, label=l) for l, c in
                  zip(["absence", "standing", "sit-by-bed", "sit-on-bed",
                       "lying-no-cover", "lying-w-cover"],
                      [colors_6[k] for k in RAW_LABELS])]
fig_ts.legend(handles=legend_patches, ncol=6, loc="upper center",
              bbox_to_anchor=(0.5, 1.02), fontsize=9)

plt.tight_layout()
ts_path = PREDICTIONS_PATH.parent / "gt_pred_timeseries.pdf"
fig_ts.savefig(ts_path, dpi=150)
print(f"Time series saved → {ts_path}")
plt.close(fig_ts)

# ── Per-class Metrics Bar Chart ────────────────────────────────────────────────
fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
x = np.arange(NUM_CLASSES)
width = 0.25
bars1 = ax_bar.bar(x - width, prec, width, label="Precision", color="tab:blue")
bars2 = ax_bar.bar(x, rec, width, label="Recall", color="tab:orange")
bars3 = ax_bar.bar(x + width, f1, width, label="F1", color="tab:green")

ax_bar.set_xlabel("Class", fontsize=12)
ax_bar.set_ylabel("Score", fontsize=12)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(LABEL_NAMES, rotation=20, ha="right")
ax_bar.set_ylim(0, 1.05)
ax_bar.legend()
ax_bar.set_title(f"Per-Class Metrics (accuracy={overall_acc:.3f})", fontsize=13)
ax_bar.grid(True, axis="y", alpha=0.3)

for bar in (*bars1, *bars2, *bars3):
    h = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
bar_path = PREDICTIONS_PATH.parent / "per_class_metrics.pdf"
fig_bar.savefig(bar_path, dpi=150)
print(f"Bar chart saved → {bar_path}")
plt.close(fig_bar)

print("\nDone.")
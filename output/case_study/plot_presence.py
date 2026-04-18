"""
Binary presence plotting script for case_study results.
Reads predictions.json (dict or list) and the test dataset ground truth,
then plots GT vs. Predicted binary presence over time.

Generates:
  - Binary presence time series (GT and Pred overlaid)
  - Per-frame accuracy (correct / wrong)
"""
import json, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKSPACE = Path("/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace")
PREDICTIONS_PATH = WORKSPACE / "output/case_study/pipeline_5class/predictions.json"
TEST_PATH = WORKSPACE / "data" / "office1_0"

sys.path.insert(0, str(WORKSPACE / "src"))
from dataset.dataset import ThermalDataset

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
all_gt = test_ds.annotations_expanded[:num_preds]
print(f"Loaded {len(all_gt)} ground-truth labels from {TEST_PATH.name}")

# ── Binary presence: label != 0 → present (1), label == 0 → absent (0) ─────────
gt_present = np.array([(g != 0) for g in all_gt], dtype=int)
pr_present = np.array([(p != 0) for p in y_pred], dtype=int)

correct = gt_present == pr_present
overall_acc = correct.mean()
print(f"Binary presence accuracy: {overall_acc:.4f}")

# per-class breakdown (present = labels 2-6)
labels_6 = [0, 2, 3, 4, 5, 6]
label_names = ["absence", "standing", "sit-by-bed", "sit-on-bed", "lying-no-cover", "lying-w-cover"]

# ── Plot: Binary presence time series ──────────────────────────────────────────
frames = np.arange(num_preds)

fig, axes = plt.subplots(2, 1, figsize=(18, 6), sharex=True)

# GT presence (step for clarity)
axes[0].step(frames, gt_present, where="post", linewidth=0.8, color="tab:blue")
axes[0].fill_between(frames, gt_present, step="post", alpha=0.2, color="tab:blue")
axes[0].set_ylabel("GT Presence", fontsize=12)
axes[0].set_yticks([0, 1])
axes[0].set_yticklabels(["absent", "present"])
axes[0].set_ylim(-0.05, 1.15)
axes[0].set_title(f"Ground Truth Binary Presence — {TEST_PATH.name}", fontsize=13)
axes[0].grid(True, alpha=0.3)

# Pred presence (step for clarity)
axes[1].step(frames, pr_present, where="post", linewidth=0.8, color="tab:orange")
axes[1].fill_between(frames, pr_present, step="post", alpha=0.2, color="tab:orange")
axes[1].set_ylabel("Pred Presence", fontsize=12)
axes[1].set_yticks([0, 1])
axes[1].set_yticklabels(["absent", "present"])
axes[1].set_ylim(-0.05, 1.15)
axes[1].set_title(f"Predicted Binary Presence — {PREDICTIONS_PATH.name}", fontsize=13)
axes[1].set_xlabel("Frame Index", fontsize=12)
axes[1].grid(True, alpha=0.3)

fig.suptitle(f"Binary Presence Over Time  (accuracy={overall_acc:.4f})", fontsize=14, y=1.01)
plt.tight_layout()
ts_path = PREDICTIONS_PATH.parent / "presence_timeseries.pdf"
fig.savefig(ts_path, dpi=150)
print(f"Binary presence time series saved → {ts_path}")
plt.close(fig)

# ── Plot: Overlay GT and Pred + mark errors ─────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(18, 4))

# correct predictions in green
ax2.scatter(frames[correct], pr_present[correct],
            c="tab:green", s=4, marker="|", label=f"correct ({correct.sum()})", zorder=2)
# wrong predictions in red
ax2.scatter(frames[~correct], pr_present[~correct],
            c="tab:red", s=12, marker="|", label=f"wrong ({(~correct).sum()})", zorder=3)

# GT as grey step behind
ax2.step(frames, gt_present, where="post", linewidth=0.6, color="grey", alpha=0.5, label="GT")

ax2.set_yticks([0, 1])
ax2.set_yticklabels(["absent", "present"])
ax2.set_ylim(-0.1, 1.2)
ax2.set_xlabel("Frame Index", fontsize=12)
ax2.set_ylabel("Presence", fontsize=12)
ax2.set_title(f"Binary Presence — GT (grey) vs. Pred (green/red)  acc={overall_acc:.4f}", fontsize=13)
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

overlay_path = PREDICTIONS_PATH.parent / "presence_overlay.pdf"
fig2.savefig(overlay_path, dpi=150)
print(f"Overlay plot saved → {overlay_path}")
plt.close(fig2)

# ── Summary report ──────────────────────────────────────────────────────────────
from sklearn.metrics import precision_score, recall_score, f1_score

prec = precision_score(gt_present, pr_present)
rec  = recall_score(gt_present, pr_present)
f1   = f1_score(gt_present, pr_present)

print(f"\n=== Binary Presence Metrics ===")
print(f"  Accuracy : {overall_acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1 Score : {f1:.4f}")

# per-class presence metrics
for label, name in zip(labels_6[1:], label_names[1:]):
    gt_c = (np.array(all_gt) == label).astype(int)
    pr_c = (np.array(y_pred) == label).astype(int)
    acc_c = (gt_c == pr_c).mean()
    print(f"  {name:<20} acc={acc_c:.4f}")

print("\nDone.")
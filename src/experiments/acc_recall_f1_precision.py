"""
Summarise accuracy, precision, recall and F1 for all baseline experiments
by loading existing output/metrics.json files.

Experiments compared:
  1. Baseline CNN (single-dataset: office0_4 → office1_0)   — output/case_study/CNN/metrics.json
  2. Baseline CNN (multi-dataset: office0_* → office1_0)       — output/case_study/CNN/metrics.json (new run)
  3. Full Kalman+tracking pipeline + 5-class CNN              — output/case_study/pipeline_5class/metrics.json
  4. LSTM baseline (office0_4 → office1_0)                     — output/LSTM/metrics.json

Note: Metrics files 1 and 4 above refer to the OLD single-dataset runs.
      The new multi-dataset CNN run writes to the same path and will overwrite them.
      Check the git status timestamps to know which is which.
"""
import sys, json
from pathlib import Path

WORKSPACE = Path("/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace")
OUTPUT_DIR = WORKSPACE / "output"

CLASS_NAMES = ["absence", "standing", "sit-by-bed", "sit-on-bed", "lying-no-cover", "lying-w-cover"]
RAW_LABELS  = [0, 2, 3, 4, 5, 6]

# ── helpers ──────────────────────────────────────────────────────────────────
def load_json(path):
    with open(path) as f:
        return json.load(f)

def load_all_metrics():
    """Load metrics from all available experiment directories."""
    paths = {
        "Baseline CNN (single-dataset)\noffice0_4 → office1_0":
            OUTPUT_DIR / "case_study" / "CNN" / "metrics.json",

        "Full Kalman+Tracking Pipeline\n(5-class CNN posture + residual classifier)":
            OUTPUT_DIR / "case_study" / "pipeline_5class" / "metrics.json",

        "Baseline LSTM (single-dataset)\noffice0_4 → office1_0":
            OUTPUT_DIR / "LSTM" / "metrics.json",
    }
    results = {}
    for label, path in paths.items():
        if path.exists():
            results[label] = load_json(path)
        else:
            print(f"[WARN] Not found: {path}")
    return results

def fmt(v, decimals=4):
    return f"{v:.{decimals}f}"

def print_divider(char="─", width=120):
    print(char * width)

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    results = load_all_metrics()

    if not results:
        print("No metrics.json files found.")
        return

    print_divider()
    print("EXPERIMENT RESULTS SUMMARY")
    print("Precision / Recall / F1 per class + overall accuracy")
    print_divider()

    for exp_label, m in results.items():
        print(f"\n{'='*120}")
        print(f"  {exp_label}")
        print(f"{'='*120}")

        # Overall accuracy
        overall_acc = m.get("overall_accuracy", None)
        if overall_acc is not None:
            print(f"\n  Overall Frame-Level Accuracy: {fmt(overall_acc)}")

        # Presence metrics (pipeline only)
        if "presence" in m:
            p = m["presence"]
            print(f"  Presence Detection (binary)  — P: {fmt(p['precision'])}  "
                  f"R: {fmt(p['recall'])}  F1: {fmt(p['f1'])}")

        # Per-class metrics
        print(f"\n  {'Class':<22} {'Precision':>12} {'Recall':>12} {'F1':>12}")
        print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*12}")

        # Determine which per-class key was used
        if "per_class_6" in m:
            per_class = m["per_class_6"]
        elif "per_class" in m:
            per_class = m["per_class"]
        else:
            per_class = None

        if per_class:
            for raw in RAW_LABELS:
                name = CLASS_NAMES[RAW_LABELS.index(raw)]
                if str(raw) in per_class:
                    d = per_class[str(raw)]
                    print(f"  {name:<22} {fmt(d['precision']):>12} {fmt(d['recall']):>12} {fmt(d['f1']):>12}")

        # Weighted / Macro averages
        if "weighted_avg" in m:
            wa = m["weighted_avg"]
            print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*12}")
            print(f"  {'Weighted Avg':<22} {fmt(wa['precision']):>12} {fmt(wa['recall']):>12} {fmt(wa['f1']):>12}")

        if "macro_avg" in m:
            ma = m["macro_avg"]
            print(f"  {'Macro Avg':<22} {fmt(ma['precision']):>12} {fmt(ma['recall']):>12} {fmt(ma['f1']):>12}")

    print_divider()
    print("\nDone.")

if __name__ == "__main__":
    main()
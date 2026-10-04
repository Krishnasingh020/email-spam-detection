import json
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
METRICS_PATH = RESULTS_DIR / "metrics.json"
TABLE_PATH = RESULTS_DIR / "comparison_table.csv"
CHART_PATH = RESULTS_DIR / "confusion_matrices.png"

def evaluate_models(trained_models, train_times, test_data):
    """
    Evaluates each model on the held-out test split.
    test_data is a tuple: (X_test_vec, y_test)
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    X_test_vec, y_test = test_data

    metrics_output = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_samples_count": len(y_test),
        "models": {}
    }
    table_rows = []

    num_test_samples = len(y_test)

    # Prepare subplot for confusion matrices (1 row, 3 columns)
    fig, axes = plt.subplots(1, len(trained_models), figsize=(5 * len(trained_models), 4.5))
    if len(trained_models) == 1:
        axes = [axes]

    best_model_name = None
    best_f1 = -1.0

    for idx, (name, model) in enumerate(trained_models.items()):
        # Measure prediction / inference time
        start_inf = time.perf_counter()
        y_pred = model.predict(X_test_vec)
        total_inf_time = time.perf_counter() - start_inf
        inf_per_sample_ms = (total_inf_time / num_test_samples) * 1000.0

        # Calculate metrics (positive label is 'spam')
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, pos_label="spam", zero_division=0))
        rec = float(recall_score(y_test, y_pred, pos_label="spam", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, pos_label="spam", zero_division=0))
        cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"]).tolist()

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

        metrics_output["models"][name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "train_time_seconds": round(train_times.get(name, 0.0), 4),
            "inference_time_per_sample_ms": round(inf_per_sample_ms, 4),
            "confusion_matrix": cm  # [[TN (ham->ham), FP (ham->spam)], [FN (spam->ham), TP (spam->spam)]]
        }

        table_rows.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "Train Time (s)": round(train_times.get(name, 0.0), 4),
            "Inference Time (ms/sample)": round(inf_per_sample_ms, 4)
        })

        # Plot confusion matrix
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(y_test, y_pred, labels=["ham", "spam"]),
            display_labels=["ham", "spam"]
        )
        disp.plot(ax=axes[idx], cmap="Blues", colorbar=False)
        axes[idx].set_title(f"{name}\n(F1: {f1:.4f})", fontsize=11, fontweight="bold")

    metrics_output["best_model"] = best_model_name

    # Save metrics.json
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_output, f, indent=4)
    print(f"[Evaluate] Saved metrics to {METRICS_PATH}")

    # Save comparison_table.csv
    df_table = pd.DataFrame(table_rows)
    df_table.to_csv(TABLE_PATH, index=False)
    print(f"[Evaluate] Saved comparison table to {TABLE_PATH}")

    # Save confusion matrices plot
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=200)
    plt.close()
    print(f"[Evaluate] Saved confusion matrices chart to {CHART_PATH}")

    # Print summary table to console
    print("\n" + "="*80)
    print("MODEL COMPARISON TABLE (Held-Out Test Set)")
    print("="*80)
    print(df_table.to_string(index=False))
    print("="*80)
    print(f"Selected Best Model: {best_model_name} (F1-score: {best_f1:.4f})\n")

    return metrics_output, df_table

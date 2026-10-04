import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import load_data
from src.train import train_all
from src.evaluate import evaluate_models

def main():
    print("\n" + "="*80)
    print("EMAIL SPAM DETECTION SYSTEM — PIPELINE EXECUTION")
    print("Project: Review-2 Demo (Comparative Study of ML Classification Algorithms)")
    print("Institution: ITS Engineering College, Greater Noida (AKTU)")
    print("="*80 + "\n")
    
    start_total = time.perf_counter()

    # Step 1: Data Acquisition
    print(">>> STEP 1: Acquiring & validating dataset...")
    df = load_data()
    print(f"Dataset ready with {len(df)} records. Class counts:")
    print(df["label"].value_counts().to_string())
    print("-" * 50)

    # Step 2: Preprocess, Feature Extract & Train
    print(">>> STEP 2: Preprocessing, Vectorizing & Training Models...")
    trained_models, vectorizer, train_times, train_data, val_data, test_data = train_all()
    print("-" * 50)

    # Step 3: Evaluate on Held-out Test Set
    print(">>> STEP 3: Evaluating models on test split...")
    metrics, df_table = evaluate_models(trained_models, train_times, test_data)
    print("-" * 50)

    elapsed_total = time.perf_counter() - start_total
    print(f">>> All steps completed successfully in {elapsed_total:.2f} seconds.")
    print("Artifacts generated:")
    print(" - Models:       models/*.joblib, models/vectorizer.joblib")
    print(" - Metrics:      results/metrics.json")
    print(" - Comparison:   results/comparison_table.csv")
    print(" - Visuals:      results/confusion_matrices.png")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

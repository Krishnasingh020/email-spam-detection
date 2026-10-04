import time
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import load_data
from src.preprocessing import clean_text
from src.features import fit_vectorizer

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

def prepare_splits(df, test_size=0.15, val_size=0.15, random_state=42):
    """
    Cleans text and creates 70% train / 15% val / 15% test stratified splits.
    """
    print("[Train] Cleaning text with preprocessing pipeline...")
    df = df.copy()
    df["cleaned_text"] = df["text"].apply(clean_text)
    
    # Filter out any that became empty after cleaning
    df = df[df["cleaned_text"].str.strip() != ""]

    X = df["cleaned_text"]
    y = df["label"]

    # First split off test set (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Next split remaining 85% into train (70/85 ~= 0.8235) and val (15/85 ~= 0.1765)
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, stratify=y_temp, random_state=random_state
    )

    print(f"[Train] Split complete -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_all():
    """
    Loads data, fits vectorizer on train split, trains the 3 classical ML models,
    records training times, and saves all models to models/*.joblib.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_splits(df)

    print("[Train] Fitting TF-IDF Vectorizer on training set...")
    vectorizer = fit_vectorizer(X_train)

    X_train_vec = vectorizer.transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "multinomial_nb": MultinomialNB(),
        "linear_svm": SVC(kernel="linear", probability=True, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    }

    trained_models = {}
    train_times = {}

    for name, model in models.items():
        print(f"[Train] Training {name}...")
        start_t = time.perf_counter()
        model.fit(X_train_vec, y_train)
        duration = time.perf_counter() - start_t
        train_times[name] = duration

        model_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, model_path)
        trained_models[name] = model
        print(f"[Train] Saved {name} to {model_path} (Training time: {duration:.2f}s)")

    return trained_models, vectorizer, train_times, (X_train_vec, y_train), (X_val_vec, y_val), (X_test_vec, y_test)

if __name__ == "__main__":
    train_all()

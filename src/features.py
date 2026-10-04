import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.joblib"

def get_vectorizer(max_features: int = 20000) -> TfidfVectorizer:
    """Returns an uninitialized TfidfVectorizer configured according to project spec."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

def fit_vectorizer(train_texts, max_features: int = 20000) -> TfidfVectorizer:
    """Fits TfidfVectorizer on training texts and saves to models/vectorizer.joblib."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    vec = get_vectorizer(max_features=max_features)
    vec.fit(train_texts)
    joblib.dump(vec, VECTORIZER_PATH)
    return vec

def load_vectorizer(path: str | Path = VECTORIZER_PATH) -> TfidfVectorizer:
    """Loads saved vectorizer from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Vectorizer file not found at {path}. Run train.py first.")
    return joblib.load(path)

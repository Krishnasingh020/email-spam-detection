import pytest
from pathlib import Path
import joblib

from src.preprocessing import clean_text
from src.features import load_vectorizer

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"

def test_clean_text_basic():
    raw = "<b>CONGRATULATIONS!</b> You won a free prize. Visit http://spam.com now!!!"
    cleaned = clean_text(raw)
    assert "congratul" in cleaned
    assert "free" in cleaned
    assert "prize" in cleaned
    assert "visit" in cleaned
    assert "http" not in cleaned
    assert "<b>" not in cleaned
    assert "!" not in cleaned

def test_clean_text_empty_and_noise():
    assert clean_text("") == ""
    assert clean_text("123 4567 890") == ""
    assert clean_text(None) == ""

def test_vectorizer_shape():
    vectorizer = load_vectorizer()
    sample_text = ["urgent project report attachment meeting"]
    transformed = vectorizer.transform(sample_text)
    assert transformed.shape[0] == 1
    assert transformed.shape[1] > 0
    assert transformed.nnz > 0

def test_model_predictions():
    vectorizer = load_vectorizer()
    model_path = MODELS_DIR / "linear_svm.joblib"
    assert model_path.exists(), "Trained model does not exist. Run run_all.py first."
    model = joblib.load(model_path)

    # Obvious spam test
    spam_sample = "WINNER! You have won a guaranteed cash prize of $10,000. Claim your reward immediately by clicking here!"
    spam_cleaned = clean_text(spam_sample)
    spam_vec = vectorizer.transform([spam_cleaned])
    pred_spam = model.predict(spam_vec)[0]
    assert pred_spam == "spam"

    # Obvious ham test
    ham_sample = "Hi team, please find attached the agenda for tomorrow's engineering sync at 10 AM. Let me know if you have questions."
    ham_cleaned = clean_text(ham_sample)
    ham_vec = vectorizer.transform([ham_cleaned])
    pred_ham = model.predict(ham_vec)[0]
    assert pred_ham == "ham"

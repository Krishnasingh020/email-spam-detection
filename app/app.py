import sys
import json
import joblib
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# Add root directory to python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing import clean_text
from src.features import load_vectorizer

app = Flask(__name__)

MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
METRICS_PATH = RESULTS_DIR / "metrics.json"

# Global model and vectorizer holders
vectorizer = None
best_model = None
best_model_name = "linear_svm"
best_model_f1 = 0.0

def load_artifacts():
    global vectorizer, best_model, best_model_name, best_model_f1
    
    # 1. Load Vectorizer
    vectorizer = load_vectorizer()

    # 2. Determine best model from metrics.json
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r") as f:
                metrics_data = json.load(f)
                best_model_name = metrics_data.get("best_model", "linear_svm")
                best_model_f1 = metrics_data.get("models", {}).get(best_model_name, {}).get("f1_score", 0.0)
        except Exception as e:
            print(f"[App] Could not read metrics.json, defaulting to linear_svm: {e}")

    # 3. Load Model
    model_path = MODELS_DIR / f"{best_model_name}.joblib"
    if not model_path.exists():
        # Fallback to any joblib in models/
        available = list(MODELS_DIR.glob("*.joblib"))
        model_files = [f for f in available if f.stem != "vectorizer"]
        if model_files:
            model_path = model_files[0]
            best_model_name = model_path.stem
        else:
            raise FileNotFoundError("No trained model found. Please run run_all.py first.")

    best_model = joblib.load(model_path)
    print(f"[App] Successfully loaded {best_model_name} from {model_path} (F1: {best_model_f1:.4f})")

def classify_text(raw_text: str):
    cleaned = clean_text(raw_text)
    token_count = len(cleaned.split()) if cleaned else 0
    
    # Handle empty/whitespace input
    if not cleaned.strip():
        return {
            "label": "ham",
            "confidence": 50.0,
            "raw_len": len(raw_text),
            "token_count": 0,
            "cleaned_text": ""
        }

    vec = vectorizer.transform([cleaned])
    pred = best_model.predict(vec)[0]

    confidence = 95.0
    if hasattr(best_model, "predict_proba"):
        probs = best_model.predict_proba(vec)[0]
        classes = list(best_model.classes_)
        if pred in classes:
            prob_val = probs[classes.index(pred)]
            confidence = round(prob_val * 100, 1)

    return {
        "label": str(pred).lower(),
        "confidence": confidence,
        "raw_len": len(raw_text),
        "token_count": token_count,
        "cleaned_text": cleaned
    }

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    email_text = ""
    if request.method == "POST":
        email_text = request.form.get("email_text", "")
        if email_text:
            result = classify_text(email_text)

    return render_template(
        "index.html",
        model_name=best_model_name.replace("_", " ").title(),
        f1_score=f"{best_model_f1:.4f}",
        result=result,
        email_text=email_text
    )

@app.route("/api/classify", methods=["POST"])
def api_classify():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Missing 'text' in request body"}), 400
    res = classify_text(text)
    return jsonify(res)

# Load artifacts upon module import so Flask test client works immediately
load_artifacts()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

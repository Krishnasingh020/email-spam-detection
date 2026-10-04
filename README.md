# Email Spam Detection System
**A Comparative Study of Classification Algorithms — Review-2 Checkpoint**

**Institution:** ITS Engineering College, Greater Noida (AKTU)  
**Department:** B.Tech Computer Science & Engineering (Semester VII)  
**Team Members:** Krishna Singh, Javed Khan, Aman Kumar Singh, Arsh Aazam  

---

## 1. Project Overview
This project performs binary text classification to accurately categorize emails/messages into **Spam** or **Ham** (legitimate). The system prioritizes minimizing False Positives (legitimate emails wrongly flagged as spam) while maintaining an overall test accuracy >95%.

The pipeline benchmarks three classical machine learning algorithms on TF-IDF features:
1. **Multinomial Naive Bayes (MNB)**
2. **Linear Support Vector Machine (Linear SVM)**
3. **Random Forest (RF, 200 estimators)**

---

## 2. Directory Structure
```
email-spam-detection/
├── data/
│   └── raw/                  # Acquired or fallback dataset (dataset.csv)
├── src/
│   ├── data_loader.py        # 3-tier fallback dataset acquisition
│   ├── preprocessing.py      # clean_text (HTML strip, URLs, NLTK stopwords, PorterStemmer)
│   ├── features.py           # TF-IDF vectorizer fit, transform, save, load
│   ├── train.py              # 70/15/15 stratified train/val/test splits & model training
│   └── evaluate.py           # Metrics computation, comparison table, confusion matrices
├── models/                   # Saved .joblib model files & vectorizer.joblib
├── results/
│   ├── metrics.json          # Complete JSON metrics per model
│   ├── comparison_table.csv  # Accuracy, precision, recall, F1, latency table
│   └── confusion_matrices.png# Confusion matrix visual chart
├── app/
│   ├── app.py                # Flask demo application
│   ├── templates/index.html  # Demo UI with quick sample buttons
│   └── static/style.css      # Academic styling with #C00000 crimson accent
├── tests/
│   ├── test_pipeline.py      # Automated pytest unit & integration tests
│   └── test_cases.md         # 10 verified test cases with live outputs
├── requirements.txt          # Python dependencies
├── run_all.py                # One-command full pipeline execution
└── README.md
```

---

## 3. Setup Instructions

```bash
# 1. Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 4. How to Run

### Run the Full Machine Learning Pipeline
Acquires data, preprocesses, vectorizes, trains all 3 models, evaluates them, and writes results:
```bash
python run_all.py
```
*(Runs in under 10 seconds)*

### Run Automated Tests
```bash
pytest -v tests/test_pipeline.py
```

### Launch the Interactive Web Demo
```bash
python app/app.py
```
Open your browser at **http://localhost:5000** to test sample emails or paste custom text.

# System Architecture & Workflow

**Project Title:** Email Spam Detection System (Using Machine Learning — A Comparative Study of Classification Algorithms)  
**Institution:** ITS Engineering College, Greater Noida (AKTU)  
**Department:** B.Tech Computer Science & Engineering (Semester VII)  
**Team Members:** Krishna Singh, Javed Khan, Aman Kumar Singh, Arsh Aazam  

---

## 1. High-Level System Architecture

The project is structured into three decoupled layers: **Data & Training Pipeline**, **Model Artifacts & Metrics Storage**, and **Inference / Presentation Layer**.

```
+-------------------------------------------------------------------------------+
|                            1. DATA & TRAINING PIPELINE                        |
|                                                                               |
|  [Tier 1: Kaggle] --(fail)--> [Tier 2: GitHub Mirror] --(fail)--> [Synthetic] |
|                                      |                                        |
|                               (dataset.csv)                                   |
|                                      v                                        |
|                          Text Preprocessing Pipeline                          |
|                       (clean_text: HTML, URLs, Stopwords, Stemming)            |
|                                      v                                        |
|                             Stratified 70/15/15 Split                         |
|                           (Train / Validation / Test)                         |
|                                      v                                        |
|                       TF-IDF Feature Extraction (Fit on Train)                |
|                                      v                                        |
|                      Train Classical Classifiers:                             |
|              - Multinomial Naive Bayes                                        |
|              - Linear Support Vector Machine (Linear SVM)                     |
|              - Random Forest (200 Trees)                                      |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                        2. ARTIFACTS & EVALUATION LAYER                        |
|                                                                               |
|   models/                              results/                               |
|   ├── vectorizer.joblib                ├── metrics.json                       |
|   ├── linear_svm.joblib                ├── comparison_table.csv               |
|   ├── multinomial_nb.joblib            └── confusion_matrices.png             |
|   └── random_forest.joblib                                                    |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                          3. INFERENCE & DEMO LAYER                            |
|                                                                               |
|            User Input (Web Form / API Request at http://localhost:5000)       |
|                                      |                                        |
|                                      v                                        |
|                  Shared clean_text() Preprocessing Function                   |
|                                      v                                        |
|                         vectorizer.transform()                                |
|                                      v                                        |
|                   best_model.predict() & predict_proba()                      |
|                                      v                                        |
|               Instant Verdict: SPAM / HAM + Confidence Percentage             |
+-------------------------------------------------------------------------------+
```

---

## 2. Component-by-Component Walkthrough

### 2.1 Data Ingestion (`src/data_loader.py`)
To prevent the project from ever breaking due to network failures or missing credentials during an evaluation, the data loader implements a **3-tier resilient fallback chain**:
1. **Tier 1 (Kaggle via `kagglehub`):** Checks for pre-configured Kaggle API keys to fetch raw Enron datasets. If credentials are not present, it immediately skips without blocking.
2. **Tier 2 (Direct GitHub Raw Mirror):** Downloads verified public datasets (such as the SMS/Email Spam Collection) using HTTP requests with a 10-second timeout.
3. **Tier 3 (Programmatic Synthetic Generator):** If completely offline, it programmatically synthesizes 2,400+ realistic emails using domain-specific vocabulary banks (lottery, urgency, financial terms for spam; project agendas, attachments, and meeting schedules for ham).

### 2.2 Text Preprocessing (`src/preprocessing.py`)
Machine learning models cannot process raw natural language strings directly. The `clean_text(text: str)` function runs sequential cleaning steps:
1. **Case Normalization:** Converts all characters to lowercase.
2. **HTML Stripping:** Decodes HTML entities and strips tags (`<p>`, `<b>`, etc.).
3. **URL Stripping:** Removes `http://` and `www.` web addresses.
4. **Punctuation & Digit Filtering:** Strips non-alphabetic noise to focus on core semantic tokens.
5. **Tokenization:** Breaks the clean string into individual word tokens.
6. **Stopword Elimination:** Filters out high-frequency English stopwords (e.g., "is", "the", "and") using NLTK's English stopword lexicon.
7. **Stemming (Porter Stemmer):** Reduces inflectional forms to their base stems (e.g., "winning", "wins", "winner" $\rightarrow$ "win").

> **Important:** The exact same `clean_text()` function is imported both during batch training and during real-time web inference in the Flask app. This prevents **train/serve skew**.

### 2.3 Feature Extraction (`src/features.py`)
Converts stemmed text strings into numerical vectors using **Term Frequency-Inverse Document Frequency (TF-IDF)**:
* `max_features=20000`: Caps vocabulary size to the top 20,000 informative terms.
* `ngram_range=(1, 2)`: Captures both unigrams ("urgent") and bigrams ("act now", "claim prize").
* `min_df=2`, `max_df=0.95`: Eliminates rare typos and words that appear in over 95% of documents.
* `sublinear_tf=True`: Dampens the effect of repetitive keywords using logarithmic scaling ($1 + \log(\text{tf})$).
* **Fit-Transform Discipline:** The vectorizer is fitted **strictly on the 70% training split**. The validation and test sets are only transformed, preventing data leakage.

### 2.4 Model Training (`src/train.py`)
Performs a stratified 70% train / 15% validation / 15% test split, preserving class balance across subsets. It trains three classical algorithms:
1. **Multinomial Naive Bayes:** Probabilistic classifier applying Bayes' Theorem with strong independence assumptions.
2. **Linear SVM (`SVC(kernel="linear", probability=True)`):** Finds the optimal maximum-margin hyperplane separating spam from legitimate messages in high-dimensional TF-IDF space.
3. **Random Forest Classifier (200 trees):** Ensemble of decision trees using bagging and random feature subspaces.

All trained models and the fitted vectorizer are serialized to the `models/` directory using `joblib`.

### 2.5 Model Evaluation (`src/evaluate.py`)
Evaluates all models on the unseen 15% test split (774 emails):
* Calculates **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
* In spam classification, **Precision** is prioritized alongside accuracy to ensure legitimate emails (Ham) are not falsely routed to spam (minimizing False Positives).
* Measures training time and per-sample inference latency (in milliseconds).
* Computes confusion matrices and renders a side-by-side comparative graphic saved to `results/confusion_matrices.png`.
* Exports machine-readable metrics to `results/metrics.json` and human-readable CSV to `results/comparison_table.csv`.

### 2.6 Demonstration Web Application (`app/app.py`)
A lightweight, responsive Flask application:
* Loads the saved vectorizer and automatically selects the top-performing model from `results/metrics.json` at startup.
* Provides a web interface with preloaded quick-test sample buttons (Spam, Business Ham, and Urgent Ham).
* Exposes both a standard web form and a `/api/classify` JSON REST endpoint.
* Returns an immediate classification badge (`SPAM` or `HAM`) accompanied by a confidence percentage bar.

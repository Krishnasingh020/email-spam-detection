# Review-2 Viva Preparation: Questions & Answers

This document lists the most common viva voce questions asked by examination panels during Review-2 project evaluations, paired with concise, technically sound answers tailored to this project.

---

### Q1: What is the primary objective of your project?
**Answer:**  
Our project implements an end-to-end Email Spam Detection System utilizing Machine Learning. We conducted a comparative study of three classification algorithms—Multinomial Naive Bayes, Linear Support Vector Machine, and Random Forest—to identify the optimal classifier that achieves both high overall accuracy (>95%) and minimal False Positives to ensure legitimate emails are never lost in spam.

---

### Q2: What dataset did you use, and how many samples does it contain?
**Answer:**  
We utilized an authenticated corpus of **5,169 real-world messages** sourced from public benchmarks (Enron/SMS Spam Collection). The dataset contains **4,516 legitimate messages (87.4%)** and **653 spam messages (12.6%)**, reflecting the natural class imbalance found in real email environments. Additionally, our data ingestion module features a 3-tier fallback chain (Kaggle $\rightarrow$ GitHub raw mirror $\rightarrow$ synthetic generator) to ensure zero failure risk during pipeline execution.

---

### Q3: Why is Precision considered more critical than Recall in email spam filtering?
**Answer:**  
In spam filtering, a **False Positive** (classifying an important legitimate email as spam) is far more damaging than a **False Negative** (allowing a spam email into the inbox). If a user misses a job interview invitation or an invoice because it went to spam, significant harm occurs. Therefore, keeping Precision close to 100% while maintaining acceptable Recall is the primary design priority.

---

### Q4: Explain the steps in your text preprocessing pipeline.
**Answer:**  
Raw text passes through seven preprocessing stages:
1. **Lowercasing:** Normalizes letter cases.
2. **HTML Stripping:** Removes tags (`<p>`, `<b>`) and unescapes entities.
3. **URL Removal:** Filters out `http` and `www` links using regex.
4. **Punctuation & Digit Elimination:** Removes noise and non-alphabetic symbols.
5. **Tokenization:** Splits text into constituent words.
6. **Stopword Filtering:** Eliminates non-discriminatory English words (e.g., "the", "and") via NLTK.
7. **Stemming:** Uses the Porter Stemmer to reduce words to their morphological roots (e.g., "awarded", "awards" $\rightarrow$ "award").

---

### Q5: What is TF-IDF and why did you choose it over simple Bag-of-Words?
**Answer:**  
Bag-of-Words merely counts raw word frequency, which biases models toward long documents and frequent words. **TF-IDF** (Term Frequency-Inverse Document Frequency) scales the count by how unique the term is across the entire corpus. Words that appear everywhere receive a low weight, while distinctive words receive high weights. We also incorporated bigrams (e.g., *"click here"*, *"win cash"*) and sublinear scaling ($1 + \log(\text{TF})$) to penalize repetitive spam keyword stuffing.

---

### Q6: Why did you fit the TF-IDF vectorizer only on the training set?
**Answer:**  
Fitting the vectorizer on the entire dataset before splitting causes **data leakage**, where information from the test set subtly influences the feature vocabulary and IDF weights. By fitting strictly on the 70% training split and only transforming the validation and test sets, we ensure an honest, unbiased evaluation.

---

### Q7: Which model performed the best in your comparative study?
**Answer:**  
**Linear Support Vector Machine (Linear SVM)** achieved the best overall performance with **98.45% Accuracy** and the highest **F1-Score of 0.9375** (Precision: 95.74%, Recall: 91.84%). While Multinomial Naive Bayes achieved 100% Precision, it had lower Recall (80.61%). Linear SVM provided the most balanced performance and was selected as the default model in our demo application.

---

### Q8: What is a Confusion Matrix, and what do your numbers show?
**Answer:**  
A confusion matrix visualizes True Positives, True Negatives, False Positives, and False Negatives. On our 774 held-out test samples:
* **Linear SVM:** 672 True Negatives (correct Ham), 90 True Positives (correct Spam), 4 False Positives, and 8 False Negatives.
* **Multinomial Naive Bayes:** 676 True Negatives, 79 True Positives, 0 False Positives, and 19 False Negatives.

---

### Q9: Did you test any edge cases, and did any fail?
**Answer:**  
Yes, we tested 10 real-world scenarios:
* **Successes:** Legitimate emails containing urgency language (*"urgent response needed by EOD"*) and financial terms (*"quarterly budget report"*) correctly passed as **HAM**, avoiding false positives.
* **Failure (TC-04):** Obfuscated leetspeak spam (*"che4p v1agra with fr33 delivery"*) was misclassified as HAM because word-level tokenization treated the alphanumeric substitutions as unknown out-of-vocabulary tokens.

---

### Q10: How do you plan to resolve the leetspeak/obfuscation failure in your final phase?
**Answer:**  
For our final semester phase, we plan to implement:
1. **Character-level n-grams and subword tokenization** to capture phonetic patterns regardless of numerical substitutions.
2. **Deep Learning:** A recurrent neural network (**LSTM / Bi-LSTM**) or a lightweight transformer (DistilBERT) capable of sequence modeling and contextual semantics.

---

### Q11: How is your Flask web demo integrated with your machine learning model?
**Answer:**  
The Flask application (`app/app.py`) loads the pre-fitted TF-IDF vectorizer and the top-performing model artifact (`linear_svm.joblib`) into memory at server startup. When a user submits an email, it calls the exact same `clean_text()` function used in training to eliminate train/serve skew, vectorizes the text, generates a prediction and confidence score via `predict_proba()`, and displays the verdict with visual confidence feedback.

---

### Q12: How fast is your pipeline?
**Answer:**  
The entire pipeline (`run_all.py`)—including data acquisition, cleaning, stratified splitting, training three models, and generating all evaluation charts—completes in **under 4 seconds** on a standard multi-core laptop. Real-time inference in the web app takes **under 50 microseconds per email**.

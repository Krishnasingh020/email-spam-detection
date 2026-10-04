# Test Cases & System Validation Suite

**Project:** Email Spam Detection System  
**Institution:** ITS Engineering College, Greater Noida (AKTU)  
**Review:** Review-2 Mid-Project Checkpoint  
**Execution Status:** Generated from actual test execution against trained pipeline (`models/linear_svm.joblib`).

| Test ID | Test Category | Input Email (Summary / Excerpt) | Expected Output | Actual Output | Confidence | Result |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| TC-01 | Obvious Spam | CONGRATULATIONS! You have been selected to win a free $1,000 W... | SPAM | SPAM | 99.7% | **PASS** |
| TC-02 | Obvious Business Ham | Hi Javed, please find attached the agenda and slide deck for t... | HAM | HAM | 96.9% | **PASS** |
| TC-03 | Short Spam | Free cash payout! Call 1-800-CLAIM-NOW urgent! | SPAM | SPAM | 100.0% | **PASS** |
| TC-04 | Obfuscated Spam | Buy che4p v1agra and medicati0n online with fr33 delivery! Ris... | SPAM | HAM | 34.9% | **FAIL** |
| TC-05 | Urgent Language in Ham (Edge Case) | URGENT: Team, please submit your feedback on the project miles... | HAM | HAM | 95.1% | **PASS** |
| TC-06 | Email with URL (Ham) | Hey Aman, check out the documentation on the official website:... | HAM | HAM | 99.7% | **PASS** |
| TC-07 | Financial Terms in Ham | Attached is the quarterly budget review spreadsheet showing de... | HAM | HAM | 88.4% | **PASS** |
| TC-08 | Very Short / Minimal Input | Ok thanks | HAM | HAM | 100.0% | **PASS** |
| TC-09 | Lottery Scam | Dear customer, your mobile number won $500,000 in the UK Natio... | SPAM | SPAM | 100.0% | **PASS** |
| TC-10 | Long Technical Context (Ham) | Dear Committee Members, this email serves as the formal submis... | HAM | HAM | 99.2% | **PASS** |

### Observations & Edge Case Discussion
1. **Urgency & Financial Context Disambiguation:** Legitimate emails containing words like *'URGENT'*, *'budget'*, *'invoice'*, or *'EOD'* (TC-05 and TC-07) are successfully recognized as **HAM**, avoiding false-positive penalties.
2. **Obfuscation & Leetspeak Limitation (TC-04):** The model correctly identifies standard promotional spam, but fails on heavy alphanumeric obfuscations ('che4p', 'v1agra', 'medicati0n') because word-level tokenization produces out-of-vocabulary terms. This is a valuable, authentic finding to present in Review-2 as motivation for subword/character n-gram embeddings in the final phase.
3. **Short Inputs & URLs:** Normal brief messages (TC-08) and technical emails containing documentation links (TC-06) correctly classify as **HAM**.

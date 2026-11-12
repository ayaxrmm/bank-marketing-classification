# Bank Marketing - Predicting Term-Deposit Subscription

Binary classification on the UCI **Bank Marketing** dataset (41,188 clients, 20 features): will a client subscribe to a term deposit after a phone call?
The project covers the full workflow — EDA, leakage-aware preprocessing, class-imbalance handling, six models, evaluation and business insights.

**Notebook:** [`bank_marketing_classification.ipynb`](bank_marketing_classification.ipynb)

## Highlights
- **Data leakage removed:** `duration` is known only *after* the call, so it was dropped to keep the model realistic.
- **Imbalanced target (11.3% "yes")** → evaluated with ROC-AUC, PR-AUC, Precision, Recall, F1 instead of accuracy; compared class weights vs. random oversampling.
- **Leak-free pipeline:** `ColumnTransformer` + `Pipeline` (One-Hot + StandardScaler), stratified train/test split, outlier capping fitted on train only.
- **Six models:** Logistic Regression, Decision Tree, Random Forest, XGBoost, KNN, SVM; `GridSearchCV` with stratified 5-fold CV.
- **Decision-threshold tuning** using cross-validated predictions on the training set.

## Results (test set, threshold 0.5)

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| **Random Forest** | **0.808** | **0.477** | 0.501 | 0.416 | 0.629 |
| Decision Tree | 0.802 | 0.458 | 0.490 | 0.405 | 0.621 |
| Logistic Regression | 0.799 | 0.445 | 0.454 | 0.351 | 0.642 |
| KNN | 0.795 | 0.447 | 0.323 | 0.655 | 0.214 |
| XGBoost | 0.788 | 0.440 | 0.388 | 0.264 | 0.731 |
| SVM (10k-row sample) | 0.776 | 0.386 | 0.505 | 0.424 | 0.624 |

Random baseline for PR-AUC ≈ 0.113. Tuning the Random Forest threshold to 0.64 raised F1 from 0.50 to 0.52 (Precision 0.42 → 0.47).

## Key insights
- Macro-economic context (`nr.employed`, `euribor3m`, consumer confidence) is the strongest driver of subscriptions.
- Clients with a **successful previous campaign** subscribe far more often (~65% vs ~9% for new clients).
- Simple, interpretable models (Logistic Regression, a depth-7 tree) come close to the ensembles — a reasonable choice for a bank.

## How to run
```bash
git clone https://github.com/ayaxrmm/bank-marketing-classification.git
cd bank-marketing-classification
pip install -r requirements.txt
# download the data — see data/README.md
jupyter notebook bank_marketing_classification.ipynb
```

## Tech stack
Python · pandas · NumPy · scikit-learn · XGBoost · Matplotlib · seaborn

## Data
UCI Machine Learning Repository. https://doi.org/10.24432/C5K306
Download **bank-additional.zip** from the UCI repository:
https://archive.ics.uci.edu/dataset/222/bank+marketing

Unzip it and put `bank-additional-full.csv` in this folder:

```
data/bank-additional-full.csv
```


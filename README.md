# Credit Risk Analysis

A machine learning pipeline for classifying loan applicants as good or bad credit risks, built on the UCI German Credit dataset.

---

## Dataset

**Source:** [UCI Machine Learning Repository — Statlog (German Credit Data)](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))

| Property | Detail |
|---|---|
| Rows | 1,000 loan applicants |
| Features | 20 (categorical + numeric) |
| Target | 0 = Good Credit, 1 = Bad Credit |
| Class Split | 70% Good / 30% Bad |
| Download | Auto-fetched via URL in code |

No manual download needed — the script fetches the data directly.

---

## Project Structure

```
credit-risk-analysis/
│
├── credit_risk_analysis.py   # Main script (all 8 sections)
├── eda_plots.png             # EDA visualisations (auto-generated)
├── model_evaluation.png      # ROC curves + confusion matrix (auto-generated)
├── advanced_roc_comparison.png  # Final model comparison plot (auto-generated)
├── README.md
└── requirements.txt
```

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
imbalanced-learn
xgboost
```

Install all dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn xgboost
```

Or using the requirements file:

```bash
pip install -r requirements.txt
```

---

## How to Run

```bash
python credit_risk_analysis.py
```

The script runs all 8 sections sequentially and prints results to the console. Three plot files are saved automatically.

### Running in Google Colab

Paste the code into a notebook. Add this cell first to install the extra packages:

```python
!pip install imbalanced-learn xgboost -q
```

---

## Pipeline Sections

| Section | Description |
|---|---|
| 1. Load Data | Fetch UCI dataset, recode target variable |
| 2. EDA | 6-panel visualisation of distributions and default rates |
| 3. Preprocessing | Label encoding, 80/20 stratified split, SMOTE (sampling_strategy=0.6) |
| 4. Model Training | Logistic Regression, Random Forest, Gradient Boosting with 5-fold CV |
| 5. Evaluation | ROC curves, confusion matrix, feature importance |
| 6. Scorecard | Per-applicant Probability of Default (PD%) output |
| 7. Summary Table | Model comparison by CV AUC and Test AUC |
| 8. Advanced Models | Calibrated GB, XGBoost, GB + Engineered Features, final ROC comparison |

---

## Results Summary

| Model | Test AUC | Accuracy | Bad Recall |
|---|---|---|---|
| **GB + Engineered Features** ✅ | **0.8039** | **77%** | **58%** |
| Gradient Boosting | 0.7989 | 77% | 58% |
| XGBoost | 0.7974 | 73% | 62% |
| Random Forest | 0.7897 | 77% | 48% |
| Logistic Regression | 0.7863 | 78% | 60% |
| Calibrated GB | 0.7836 | 77% | 55% |

**Winner: Gradient Boosting + Engineered Features** with Test AUC of 0.8039.

---

## Key Design Decisions

### Why SMOTE with sampling_strategy=0.6?
Full 50/50 SMOTE caused CV-to-Test AUC gaps of 0.08–0.12 (overfitting). A softer 0.6 ratio reduced the gap to 0.036–0.040 while still improving Bad recall from 40–55% to 48–60%.

### Why not use the 0.40 threshold?
Threshold tuning at 0.40 only gained +4% Bad recall while dropping accuracy from 77% to 73% and worsening Bad F1 from 0.60 to 0.58. The default 0.50 threshold is more balanced.

### Why did Calibrated GB underperform?
Isotonic calibration overfitted to the SMOTE-augmented training distribution, which doesn't perfectly represent real test data. A known failure mode on datasets under 5,000 rows.

### Why is there a ~0.80 AUC ceiling?
All six models converged within a 0.020 AUC band. This is a documented property of the German Credit dataset in academic literature — the data is small (1,000 rows), from the 1990s, and lacks modern predictive features like payment history and debt-to-income ratio.

---

## Engineered Features

Three domain-informed features were added to capture repayment burden signals:

```python
monthly_burden   = credit_amount / (duration + 1)       # monthly repayment pressure
amount_per_age   = credit_amount / (age + 1)            # loan size vs applicant maturity
duration_x_rate  = duration * installment_rate          # compounded repayment obligation
```

---

## Top Risk Predictors (from Random Forest feature importance)

1. `checking_status` — no/negative balance = highest risk
2. `credit_history` — past defaults are the strongest signal
3. `duration` — longer loans = higher default probability
4. `credit_amount` — larger loans = more exposure
5. `savings` — low savings = elevated risk
6. `monthly_burden` — engineered: high monthly pressure

---

## Industry Context

| Benchmark | AUC | Bad Recall |
|---|---|---|
| Random guessing | 0.50 | ~30% |
| This project | 0.80 | 58% |
| Academic best (German Credit) | 0.82–0.85 | ~65% |
| Bank production models | 0.85–0.95 | 75%+ |

This project sits between "good academic result" and "production grade" — appropriate for a 1,000-row public dataset.

---

## Next Steps

- **Ensemble** the top 3 models for a potential +0.01–0.02 AUC gain
- **Upgrade dataset** to [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (300,000+ rows)
- **Add SHAP** explainability for per-applicant risk driver reporting
- **Cost-sensitive scoring** — weight false negatives at 5× false positives to reflect real lending costs

---

## License

Dataset: Public domain (UCI ML Repository).  
Code: MIT — free to use and modify.

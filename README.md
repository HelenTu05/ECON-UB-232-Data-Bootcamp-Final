# ✈️ Flight Delay Prediction
> Predicting U.S. domestic flight delays using machine learning on 5.8M flight records.

**Data Bootcamp Final Project** · 2015 USDOT Dataset · scikit-learn

---

## Overview

Can we predict whether a flight will arrive **15+ minutes late** using only information available before departure — airline, origin airport, scheduled hour, month, etc.?

This project builds and compares **5 binary classifiers**, extends to **multi-class delay prediction** (On Time / Short / Long / Cancelled), applies **GridSearchCV** for hyperparameter tuning, and analyzes business impact via a **profit curve**.

A key challenge throughout is **class imbalance**: 82.4% of flights are on time, causing naive models to achieve high accuracy by predicting "on time" for everything — with Recall ≈ 0%. We address this with `class_weight='balanced'` and downsampling.

---

## Results

| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 59.7% | 60.3% | 34.5% | 0.633 |
| KNN (k=5) | 56.5% | 58.9% | 32.2% | 0.597 |
| Decision Tree | 60.6% | 60.5% | 35.1% | 0.641 |
| Naive Bayes | 63.9% | 40.5% | 28.3% | 0.576 |
| **Random Forest** ⭐ | **60.3%** | **61.9%** | **35.4%** | **0.653** |

**Random Forest** achieved the best Recall and ROC-AUC overall.

**Top predictive features (Random Forest):**
1. `HOUR` — 0.361 · Evening flights (18–21h) exceed 24% delay rate
2. `AIRLINE` — 0.157 · Spirit 28% vs Hawaiian 10%
3. `MONTH` — 0.118 · June worst (21.8%), October best (12.1%)

**Lift:** Targeting the top 20% highest-risk flights captures **33.9% of all delays** — a **1.69× lift** over random selection.

**Profit curve** (TP = +$50, FP = −$10): KNN achieves the highest simulated profit at **$1.19M**, showing that ranking quality at the decision boundary matters as much as overall AUC.

**Multi-class prediction** (4 classes): Tuned Random Forest reaches **80.6% test accuracy**, though train accuracy of 99.9% indicates overfitting — a key area for future work.

---

## Dataset

**Source:** [2015 Flight Delays and Cancellations — Kaggle / USDOT](https://www.kaggle.com/datasets/usdot/flight-delays)

| File | Description |
|---|---|
| `flights.csv` | ~5.8M domestic flight records |
| `airlines.csv` | IATA code → airline name |
| `airports.csv` | IATA code → city, state, lat/lon |

> ⚠️ Data files are **not included** due to size. Download from Kaggle and place all three CSVs in the same folder as the notebook before running.

---

## Repository Structure

```
flight-delay-prediction/
│
├── flight_delay_prediction.ipynb   # Main notebook — EDA + modeling + extended analysis
├── README.md
└── requirements.txt
```

---

## How to Run

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/flight-delay-prediction.git
cd flight-delay-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data from Kaggle and place flights.csv, airlines.csv, airports.csv here

# 4. Launch notebook
jupyter notebook flight_delay_prediction.ipynb
```

---

## Notebook Structure

| Section | Content |
|---|---|
| 1. Introduction | Overview, predictive task, summary findings |
| 2. Data Loading | Load and inspect all three CSVs |
| 3. Preprocessing | Create target, drop leakage columns, handle NAs |
| 4. EDA | Delay by hour, month, airline, airport, cause, route, state |
| 5. Feature Engineering | One-hot encoding, scaling, train/test split |
| 6. Modeling | 5 pipelines with class imbalance handling |
| 7. Results | Metrics table, confusion matrices, ROC curves, lift, feature importance |
| 8. Conclusion | Summary, limitations, next steps |
| Extended | GridSearchCV, profit curve, multi-class prediction |

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

Install all:
```bash
pip install -r requirements.txt
```

---

## Key Findings

- **Departure hour is the strongest signal** — a single feature accounts for 36% of Random Forest's predictive power
- **Naive Bayes illustrates a classic pitfall** — highest accuracy (63.9%) but lowest recall (40.5%), showing why accuracy alone is misleading for imbalanced data
- **KNN outperforms on profit** despite lower AUC — highlighting the difference between ranking quality and threshold-based classification
- **Multi-class RF overfits severely** (train 99.9% vs test 80.6%) — regularization via `max_depth` is the clear next step

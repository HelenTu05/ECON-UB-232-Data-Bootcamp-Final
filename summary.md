# Executive Summary — Flight Delay Prediction

**Project:** Data Bootcamp Final Project  
**Dataset:** 2015 U.S. Domestic Flights (USDOT, 5.8M records)  
**Task:** Predict flight delays using pre-departure information only

---

## Business Objective

Flight delays cost U.S. airlines billions annually and disrupt millions of 
travelers. This project asks: given information available *before* a flight 
departs — airline, origin, scheduled hour, month — can we reliably predict 
whether it will arrive more than 15 minutes late? Accurate predictions enable 
airlines to proactively alert passengers and allocate ground resources to 
highest-risk flights.

---

## Data

We used the 2015 USDOT flight dataset (5.8M records), sampling 200,000 
flights for modeling. The dataset is heavily imbalanced: **82.4% of flights 
arrive on time**, requiring special handling to avoid models that trivially 
predict "on time" for everything.

Key features used: airline, origin/destination airport, month, day of week, 
and scheduled departure hour.

---

## Key EDA Findings

- **Departure hour** is the strongest delay signal — flights departing after 
  6pm have delay rates exceeding 24%, as delays cascade throughout the day
- **Airline matters** — Spirit Air Lines had the highest delay rate (~28%), 
  Hawaiian Airlines the lowest (~10%)
- **Seasonality** — June had the worst delay rate (21.8%); September had the best
- **Delay causes** — weather causes the longest individual delays (23.5 min 
  avg); late aircraft is the most frequent cause year-round
- **Geography** — Illinois, West Virginia, and Maryland had the highest 
  state-level delay rates (>22%); the IAH→ORD route was most delay-prone 
  at 32.6%

---

## Models and Methods

We trained **5 binary classifiers** using scikit-learn Pipelines with 
OneHot encoding and StandardScaling. Class imbalance was addressed via 
`class_weight='balanced'` (LR, DT, RF) and downsampling (KNN). We extended 
the analysis to **4-class delay prediction** and applied **GridSearchCV** 
for hyperparameter tuning.

---

## Results

**Random Forest** was the best binary classifier (ROC-AUC = 0.653, 
Recall = 61.9%). Targeting the top 20% highest-risk flights captures 
**33.9% of all delays — a 1.69× lift** over random selection.

The **profit curve** (TP = +$50, FP = −$10) shows maximum expected profit 
of ~$413K–418K for Decision Tree and Random Forest, achieved by flagging 
approximately 48–50% of flights.

For **multi-class prediction**, Random Forest achieved 80.6% test accuracy, 
though severe overfitting (train: 99.9%) points to class imbalance as the 
primary challenge — not model complexity.

---

## Conclusion and Next Steps

Pre-departure delay prediction is feasible but inherently limited without 
real-time weather data. For deployment, we recommend:

1. **Add NOAA weather data** — the single highest-impact improvement
2. **Use time-based train/test split** — train on Jan–Sep, test on Oct–Dec
3. **Address multi-class imbalance** via SMOTE or stronger regularization
4. **Benchmark XGBoost** — typically outperforms Random Forest on tabular data

# Formula 1 Race Finish Position Predictor: Project Report & Development Journey

This repository documents the developmental journey, data engineering decisions, and machine learning models built to predict the final finishing positions of Formula 1 drivers.

---

## Executive Summary

Predicting Formula 1 race outcomes is a highly complex task influenced by mechanical reliability, driver skill, track characteristics, starting grid positions, and seasonal car performance. 

This project documents a chronological progression from a basic baseline Random Forest regressor to an optimized, tuned XGBoost regressor. By systematically addressing **data leakage**, **race retirements (DNFs)**, **temporal grid size variations**, and **points scale changes**, the prediction error (**Test MAE**) was successfully reduced from **`2.38`** to **`1.95`** positions.

### Main Modeling Assumption
Currently, the dataset is filtered to include only drivers who finished the race (`statusId == 1`). Thus, the model answers the conditional question: **"Given that a driver successfully finishes the race, what will their final position be?"**

---

## Performance Summary

The table below summarizes the performance of our models across different phases of the project:

| Phase | Model | Key Features / Data Filter | Test $R^2$ | Test MAE | Train MAE |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **V1** | Random Forest (Base) | Cleaned dataset (No data leakage protection) | 0.5640 | 2.3806 | — |
| **V2** | Random Forest (Base) | Temporal split (Train: <2022, Test: $\ge$2022) | 0.5311 | 2.7320 | — |
| **V3** | Random Forest (Base) | Added rolling form features (Last 3, 5, 10 races) | 0.6079 | 2.4716 | — |
| **V3.1**| Random Forest (Base) | Added rolling grid positions | 0.6135 | 2.4605 | — |
| **V4** | Random Forest (Base) | Cleaned anomalies (Finished statusId = 1 only) | 0.6170 | 2.0917 | — |
| **V4.1**| Random Forest (Base) | Optimal training window filter (Year $\ge$ 2000) | 0.6308 | 2.0616 | 1.1385 |
| **V5** | Random Forest (Tuned)| Target Encoding Experiment (Unseen drivers = 11.0) | 0.5920 | 2.1641 | 1.1559 |
| **V6** | Random Forest (Tuned)| Normalized Points Fractions + Tuned Hyperparams | 0.6229 | 2.0288 | 1.5353 |
| **V7** | **XGBoost (Tuned)** | **Normalized Points Fractions + Tuned Hyperparams** | **0.6507** | **1.9538** | **1.7062** |

### Final Baseline Comparisons (V6 / V7 Dataset)
To validate that our models outperform simple heuristic reasoning, we compare them against four domain-specific baseline benchmarks:

*   **Mean Baseline MAE:** `3.7487`
*   **Median Baseline MAE:** `3.9448`
*   **Grid Baseline MAE:** `2.9529` (Predicting that the finishing position equals the starting grid position)
*   **Driver Form Baseline MAE:** `2.3205` (Predicting that the finishing position equals the average of the last 5 races)

The tuned **XGBoost model (Test MAE: 1.9538)** outperforms the best heuristic benchmark (Driver Form) by **15.8%** and the Grid baseline by **33.8%**.

---

## Detailed Development Journey

### V1 & V2: Establishing a Temporal Baseline
Initially, the data was randomly split into training and test sets. However, this introduced severe **data leakage** (e.g., training on 2023 data to predict a 2012 race). 

To resolve this, we enforced a strict **temporal split**:
*   **Training Set:** Races before 2022
*   **Test Set:** Races in and after 2022 (2022–2023 seasons)

Applying this split initially caused the Test MAE to increase to `2.7320` because the model had no way of knowing how team or driver performance shifted out-of-distribution in the newer seasons.

### V3: Temporal Feature Engineering (Rolling Form)
To allow the model to learn recent performance trends rather than relying solely on static historical identities, we engineered several rolling-window features:
1.  **Driver Form:** Average finishing position of the driver in the last 3, 5, and 10 races.
2.  **Constructor Form:** Average finishing position of the constructor's cars in the last 3 and 5 races.
3.  **Grid Form:** Average starting grid position of the driver/constructor over the last 3 and 5 races.
4.  **Circuit Form:** Historical average finishing position of the driver at the specific circuit.
5.  **Driver Improving Rate:** Calculated as the difference between the driver's short-term and long-term form (`last_3_avg - last_10_avg`).

Introducing these features significantly boosted the Test $R^2$ to `0.6135` and lowered the Test MAE to `2.4605`. It also shifted the model's focus from static driver/constructor identities to active local performance histories.

### V4: Incident Filtering & Data Quality Discovery
By analyzing the largest prediction errors, we made a key observation:
> **Anomalous Retrospect:** In an Austrian GP, Max Verstappen started 2nd but finished 20th due to a mechanical retirement. In the raw dataset, he was classified as 20th under `statusId = 17` (not finished).

DNFs (Did Not Finish) due to mechanical failures, collisions, or disqualifications are highly stochastic and unpredictable events. Including them introduces massive target noise, as a driver starting 2nd could retire on lap 1 through no fault of their own or their car's speed. By filtering the dataset to include only completed races (`statusId == 1`), we removed this random noise, allowing the model to focus on predicting the pure performance-based finishing positions of the cars and drivers.

**Result:** The Test MAE dropped significantly to **`2.0917`**, confirming that race-ending incidents were a primary source of prediction noise.

---

## Addressing Skewness & Regression to the Mean

### Visual Diagnostics
When plotting the predicted vs. actual finishing positions, we observed that the model's predictions were heavily constrained, rarely predicting finishes below 12th place and never predicting finishes worse than 14th.

<img src="Documentation/scatter_rf_v4.png" alt="Actual vs Predicted Scatter Plot" width="350"/>
<img src="Documentation/error_dist.png" alt="Error Distribution" width="350"/>

This behavior is a consequence of **regression to the mean** in ensemble regression models. Because tree-based regressors (like Random Forest) average the target values of samples falling into the same terminal leaf node, predictions are pulled towards the average finishing position. When combined with target label imbalance, where the training dataset is heavily skewed towards top finishes, the model behaves conservatively and avoids predicting extreme lower-midfield or backmarker positions.

<img src="Documentation/finish_v4.png" alt="Target Skewness" width="350"/>

### Historical Grid Size Variations
Further investigation revealed that grid sizes were much larger in the early years of the dataset (sometimes up to 30+ cars), whereas the modern era (post-2010) has maintained a stable grid size of 20–22 cars. High-ranking positions (1st–10th) exist in every season, but low-ranking positions (15th–30th) only appeared in older years, skewing the overall target distribution.

Prior to 2000, grid sizes in F1 varied widely, and the distribution of finishing positions was highly skewed. Since 2010, grid sizes have stabilized at 20–22 cars. Restricting our training data to a window starting in 2000 (the optimal training window) aligned the target distributions between the training and test sets. It eliminated historical noise from oversized grids and provided the model with a more consistent distribution of finishing positions, leading to much better generalization.

<img src="Documentation/finish_raw.png" alt="Raw Finish Positions Distribution" width="350"/>
<img src="Documentation/finish_2010.png" alt="Post-2010 Finish Positions Distribution" width="350"/>

---

## Advanced Feature Engineering Experiments

### 1. Target Encoding Experiment (Fail)
We attempted to replace raw categorical IDs (`driverId` and `constructorId`) with their target-encoded career average finishing positions. 
*   **The Theory:** Help the model learn driver strength without splitting on arbitrary ID integers. Unseen rookie drivers were mapped to a neutral fallback of `11.0`.
*   **The Reality:** The Test MAE increased to `2.1641` and the importance of the starting `grid` feature doubled.
*   **The Explanation:** Target encoding using career-long global averages failed because it completely washes out the highly seasonal nature of Formula 1 performance. F1 is a sport where car development cycles and team dominance change drastically across years. A driver's career average over 20 years represents a muddy mix of their champion years in dominant cars and their struggling years in uncompetitive cars. Consequently, global averages are poor proxies for current race speed. 
*   **Role of IDs in Trees:** Decision trees split on individual thresholds, allowing them to partition consistently assigned categorical identifiers (like `driverId`) into distinct subgroups. By pairing raw IDs with rolling temporal features, the model can isolate individual drivers and apply specific offsets without assuming any ordinal relationship between the IDs.

### 2. Normalized Points Fractions (Success)
A driver's current championship points total is a strong indicator of their current form. However, because F1 changed its points scoring system multiple times (e.g., 10 points for a win before 2010 vs. 25 points after 2010), raw points totals are inconsistent across decades.

We introduced two new self-normalizing features:
*   `driver_points_fraction` = $\frac{\text{Driver's cumulative points in the current season before this race}}{\text{Total points scored by all drivers in the season before this race}}$
*   `constructor_points_fraction` = $\frac{\text{Team's cumulative points in the current season before this race}}{\text{Total points scored by all drivers in the season before this race}}$

Both features default to `0.0` for Round 1 of any season. This normalized fraction is highly consistent across F1 eras.

Adding these features lowered our Random Forest Test MAE to **`2.0288`** and they immediately ranked as top-8 features in importance.

---

## Final Tuning & Model Comparison

Using **3-Fold Time-Split Cross-Validation** via `RandomizedSearchCV` on the training set, we optimized hyperparameters for both models to minimize Test MAE. Hyperparameter tuning (e.g., optimizing tree depth or minimum samples per leaf) helps the model find the optimal bias-variance tradeoff on the existing feature space, preventing overfitting. However, it cannot inject new information. Feature engineering, on the other hand, introduces crucial domain-specific knowledge (like current seasonal momentum via rolling form and normalized points fractions) that decision trees cannot synthesize on their own from raw, static variables. Therefore, feature engineering consistently yielded the largest performance gains.

### Random Forest (Tuned)
*   **Best Parameters:** `{'n_estimators': 500, 'min_samples_leaf': 10, 'max_features': 'sqrt', 'max_depth': 16}`
*   **Test MAE:** **`2.0288`** (Train MAE: `1.5353`)

### XGBoost (Tuned)
*   **Best Parameters:** `{'subsample': 0.8, 'n_estimators': 500, 'min_child_weight': 5, 'max_depth': 4, 'learning_rate': 0.03, 'colsample_bytree': 0.8}`
*   **Test MAE:** **`1.9538`** (Train MAE: `1.7062`)

XGBoost achieved the best generalization and lowest error rate by using gradient boosting to sequentially correct residual errors, achieving an average finish prediction accuracy within **1.95 positions**.

---

## Feature Importances (XGBoost Final)

The final feature importances highlight that predicting F1 finishing positions is heavily dependent on:
1.  **Starting Grid Position (`grid`)** — the single most predictive feature.
2.  **Recent Constructor & Driver Form** (`constructor_avg_pos_last_5`, `driver_avg_pos_last_10`, `driver_avg_pos_last_5`).
3.  **Championship Position Fractions** (`constructor_points_fraction`, `driver_points_fraction`).

---

## Lessons Learned

*   **Feature Engineering > Algorithmic Complexity:** The most significant performance improvements came from engineering temporal features and normalizing points across F1 eras, rather than upgrading algorithms or scaling hyperparameters.
*   **Temporal Validation is Critical:** Standard cross-validation causes target leakage when dealing with time-series or sequential sporting datasets. Embodying time-split CV structures was essential to evaluate out-of-distribution performance accurately.
*   **Data Quality Over Model Complexity:** Eliminating stochastic retirements (DNFs) and focusing on a consistent historical window (post-2000) yielded a massive drop in MAE. Cleaning the target distribution is more effective than forcing a model to learn unresolvable noise.
*   **Structured Error Analysis:** Reviewing the largest prediction anomalies directly exposed target leakage, data cleaning opportunities, and the regression-to-the-mean effect.
*   **Domain Knowledge Outperforms Blind Tuning:** Recognizing the impact of changing points systems and seasonal dominance allowed us to engineer self-normalizing fractions, which raw algorithms could not synthesize from categorical IDs alone.

---

## Future Work

*   **Separate DNF Prediction Pipeline:** Predict the probability of a driver retiring (DNF) using a binary classification model, and combine it with this regression model to produce a final expected finishing position.
*   **Incorporate Weather Data:** Integrate rain probability, air/track temperatures, and humidity, as wet weather dramatically changes race predictability.
*   **Qualifying Lap Times:** Utilize absolute and relative qualifying lap time gaps (in milliseconds) instead of just grid rank to represent raw car pace more precisely.
*   **Pit Stop Strategy:** Include average pit stop durations, tire compound selections, and tire life parameters.
*   **SHAP Explanations:** Apply SHAP (SHapley Additive exPlanations) values to explain individual race predictions and visualize feature contributions for specific drivers.
*   **Classification Targets:** Experiment with classification formulations, such as predicting a podium finish (Top 3), points finish (Top 10), or specific position brackets.

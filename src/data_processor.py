from pandas import errors
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import os

def load_data():
    races = pd.read_csv("data/raw/races.csv")
    results = pd.read_csv("data/raw/results.csv")
    status = pd.read_csv("data/raw/status.csv")
    races = races[["raceId","year","round","circuitId"]]
    results = results[["resultId","raceId","driverId","constructorId","grid","position","statusId"]]
    results = results[results["statusId"] == 1]
    data = results.merge(races, on="raceId")
    data = data.merge(status, on="statusId")
    data = data[data["position"] != "\\N"]
    data["position"] = data["position"].astype(int)
    data = data[data["year"] >= 2000]
    data = data.sort_values(by=["year","round"]).reset_index(drop=True)
    data.to_csv("data/processed/cleaned_data.csv", index=False)

def process_data():
    data = pd.read_csv("data/processed/cleaned_data.csv")
    data = temporal_features(data)
    train_data = data[data["year"] < 2022]
    test_data = data[data["year"] >= 2022]
    train_data.to_csv("data/processed/train_data.csv", index=False)
    test_data.to_csv("data/processed/test_data.csv", index=False)
    print("Data processed successfully\n")
    features_to_drop = ["position", "raceId", "resultId", "round", "year", "statusId", "status"]

    x_train = train_data.drop(columns = features_to_drop)
    y_train = train_data["position"]
    x_test = test_data.drop(columns = features_to_drop)
    y_test = test_data["position"]


    return x_train, y_train, x_test, y_test

def temporal_features(data):
    data["driver_avg_pos_last_3"] = (data.groupby("driverId")["position"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()))
    data["constructor_avg_pos_last_3"] = (data.groupby("constructorId")["position"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()))

    data["driver_avg_pos_last_5"] = (data.groupby("driverId")["position"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))
    data["constructor_avg_pos_last_5"] = (data.groupby("constructorId")["position"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))

    data["driver_avg_pos_last_10"] = (data.groupby("driverId")["position"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean()))

    data["driver_avg_grid_last_3"] = (data.groupby("driverId")["grid"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()))
    data["constructor_avg_grid_last_3"] = (data.groupby("constructorId")["grid"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()))

    data["driver_avg_grid_last_5"] = (data.groupby("driverId")["grid"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))
    data["constructor_avg_grid_last_5"] = (data.groupby("constructorId")["grid"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))

    data["driver_avg_pos_at_circuit"] = (data.groupby(["driverId", "circuitId"])["position"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean()))

    # data["driver_pos_std_last_5"] = (data.groupby("driverId")["position"]
    #     .transform(lambda x: x.shift(1).rolling(5, min_periods=1).std()))

    data["driver_avg_pos_last_3"] = data["driver_avg_pos_last_3"].fillna(11)
    data["constructor_avg_pos_last_3"] = data["constructor_avg_pos_last_3"].fillna(11)

    data["driver_avg_pos_last_5"] = data["driver_avg_pos_last_5"].fillna(11)
    data["constructor_avg_pos_last_5"] = data["constructor_avg_pos_last_5"].fillna(11)

    data["driver_avg_pos_last_10"] = data["driver_avg_pos_last_10"].fillna(11)

    data["driver_avg_grid_last_3"] = data["driver_avg_grid_last_3"].fillna(11)
    data["constructor_avg_grid_last_3"] = data["constructor_avg_grid_last_3"].fillna(11)

    data["driver_avg_grid_last_5"] = data["driver_avg_grid_last_5"].fillna(11)
    data["constructor_avg_grid_last_5"] = data["constructor_avg_grid_last_5"].fillna(11)

    data["driver_avg_pos_at_circuit"] = data["driver_avg_pos_at_circuit"].fillna(11)
    # data["driver_pos_std_last_5"] = data["driver_pos_std_last_5"].fillna(0)

    # Calculate driver improving rate (avg_last_3 - avg_last_10)
    data["driver_improving_rate"] = data["driver_avg_pos_last_3"] - data["driver_avg_pos_last_10"]

    return data

def random_forest(x_train, y_train, x_test, y_test):
    model = RandomForestRegressor(n_estimators=500,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1)
    model.fit(x_train, y_train)
    print("......Overall Test Results (Random Forest)......")
    print("test score:", model.score(x_test, y_test))
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    print("Test MAE:", mae)
    
    y_train_pred = model.predict(x_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    print("Train MAE:", train_mae, "\n")


    print("......Baseline Statistics......")

    mean_val = y_train.mean()
    y_pred_mean = np.full_like(y_test, mean_val, dtype=float)
    mae_mean = mean_absolute_error(y_test, y_pred_mean)
    print(f"Mean Baseline MAE: {mae_mean}")

    median_val = y_train.median()
    y_pred_median = np.full_like(y_test, median_val, dtype=float)
    mae_median = mean_absolute_error(y_test, y_pred_median)
    print(f"Median Baseline MAE: {mae_median}")

    y_pred_grid = x_test["grid"]
    mae_grid = mean_absolute_error(y_test, y_pred_grid)
    print(f"Grid Baseline MAE: {mae_grid}")

    y_pred_driver_form = x_test["driver_avg_pos_last_5"]
    mae_driver_form = mean_absolute_error(y_test, y_pred_driver_form)
    print(f"Driver Form Baseline MAE: {mae_driver_form}\n")

    print("......Feature Importances......")
    importance_df = pd.DataFrame({
    "feature": x_train.columns,
    "importance": model.feature_importances_
    })
    importance_df = importance_df.sort_values("importance", ascending=False)
    print(importance_df)

    errors = x_test.copy()
    errors["actual"] = y_test
    errors["predicted"] = y_pred
    errors["abs_error"] = abs(errors["actual"] - errors["predicted"])

    test_data = pd.read_csv("data/processed/test_data.csv")
    errors["year"] = test_data["year"].values
    errors["raceId"] = test_data["raceId"].values
    errors["statusId"] = test_data["statusId"].values
    errors["status"] = test_data["status"].values

    errors.sort_values("abs_error", ascending=False).head(50).to_csv("data/processed/errors.csv", index=False)
    make_plots_rf(y_test, y_pred)


def xgboost(x_train, y_train, x_test, y_test):
    model = XGBRegressor(n_estimators=500,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42)
    model.fit(x_train, y_train)
    print("......Overall Test Results (XGBoost)......")
    print("test score:", model.score(x_test, y_test))
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    print("Test MAE:", mae)
    
    y_train_pred = model.predict(x_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    print("Train MAE:", train_mae, "\n")


    print("......Baseline Statistics......")

    mean_val = y_train.mean()
    y_pred_mean = np.full_like(y_test, mean_val, dtype=float)
    mae_mean = mean_absolute_error(y_test, y_pred_mean)
    print(f"Mean Baseline MAE: {mae_mean}")

    median_val = y_train.median()
    y_pred_median = np.full_like(y_test, median_val, dtype=float)
    mae_median = mean_absolute_error(y_test, y_pred_median)
    print(f"Median Baseline MAE: {mae_median}")

    y_pred_grid = x_test["grid"]
    mae_grid = mean_absolute_error(y_test, y_pred_grid)
    print(f"Grid Baseline MAE: {mae_grid}")

    y_pred_driver_form = x_test["driver_avg_pos_last_5"]
    mae_driver_form = mean_absolute_error(y_test, y_pred_driver_form)
    print(f"Driver Form Baseline MAE: {mae_driver_form}\n")

    print("......Feature Importances......")
    importance_df = pd.DataFrame({
    "feature": x_train.columns,
    "importance": model.feature_importances_
    })
    importance_df = importance_df.sort_values("importance", ascending=False)
    print(importance_df)

    errors = x_test.copy()
    errors["actual"] = y_test
    errors["predicted"] = y_pred
    errors["abs_error"] = abs(errors["actual"] - errors["predicted"])

    test_data = pd.read_csv("data/processed/test_data.csv")
    errors["year"] = test_data["year"].values
    errors["raceId"] = test_data["raceId"].values
    errors["statusId"] = test_data["statusId"].values
    errors["status"] = test_data["status"].values

    errors.sort_values("abs_error", ascending=False).head(50).to_csv("data/processed/errors.csv", index=False)

def make_plots_rf(y_test, y_pred):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    axes[0].scatter(y_test, y_pred, alpha=0.5)
    axes[0].set_xlabel("Actual Position")
    axes[0].set_ylabel("Predicted Position")
    axes[0].set_title("Actual vs Predicted Position (Random Forest)")

    # Error distribution histogram
    errors = y_test - y_pred
    axes[1].hist(errors, bins=50)
    axes[1].set_xlabel("Error")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Error Distribution (Random Forest)")

    plt.tight_layout()
    plt.show()

    

load_data()
x_train, y_train, x_test, y_test = process_data()
print("--- RUNNING RANDOM FOREST ---")
random_forest(x_train, y_train, x_test, y_test)
# print("\n--- RUNNING XGBOOST ---")
# xgboost(x_train, y_train, x_test, y_test)
print(os.getcwd())
print(y_train.describe())


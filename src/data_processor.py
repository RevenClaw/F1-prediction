from pandas import errors
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
    results = results[["resultId","raceId","driverId","constructorId","grid","position","points","statusId"]]
    
    # Merge results and races to get year and round
    data = results.merge(races, on="raceId")
    
    # Calculate total points scored by all drivers in each round
    round_total_points = data.groupby(["year", "round"])["points"].sum().reset_index()
    round_total_points = round_total_points.sort_values(by=["year", "round"])
    round_total_points["cumulative_total_points"] = round_total_points.groupby("year")["points"].cumsum()
    round_total_points["total_points_before_round"] = round_total_points.groupby("year")["cumulative_total_points"].shift(1).fillna(0.0)
    round_total_points = round_total_points[["year", "round", "total_points_before_round"]]

    # Calculate driver points before each round in the year
    driver_round_points = data.groupby(["driverId", "year", "round"])["points"].sum().reset_index()
    driver_round_points = driver_round_points.sort_values(by=["driverId", "year", "round"])
    driver_round_points["cumulative_points"] = driver_round_points.groupby(["driverId", "year"])["points"].cumsum()
    driver_round_points["driver_points_before_race"] = driver_round_points.groupby(["driverId", "year"])["cumulative_points"].shift(1).fillna(0.0)
    driver_round_points = driver_round_points[["driverId", "year", "round", "driver_points_before_race"]]
    
    # Calculate constructor points before each round in the year
    constructor_round_points = data.groupby(["constructorId", "year", "round"])["points"].sum().reset_index()
    constructor_round_points = constructor_round_points.sort_values(by=["constructorId", "year", "round"])
    constructor_round_points["cumulative_points"] = constructor_round_points.groupby(["constructorId", "year"])["points"].cumsum()
    constructor_round_points["constructor_points_before_race"] = constructor_round_points.groupby(["constructorId", "year"])["cumulative_points"].shift(1).fillna(0.0)
    constructor_round_points = constructor_round_points[["constructorId", "year", "round", "constructor_points_before_race"]]
    
    # Merge cumulative points and total points back into data
    data = data.merge(driver_round_points, on=["driverId", "year", "round"], how="left")
    data = data.merge(constructor_round_points, on=["constructorId", "year", "round"], how="left")
    data = data.merge(round_total_points, on=["year", "round"], how="left")

    # Normalize to fraction of points allotted (preventing division by zero in round 1)
    data["driver_points_fraction"] = np.where(
        data["total_points_before_round"] > 0,
        data["driver_points_before_race"] / data["total_points_before_round"],
        0.0
    )
    data["constructor_points_fraction"] = np.where(
        data["total_points_before_round"] > 0,
        data["constructor_points_before_race"] / data["total_points_before_round"],
        0.0
    )

    # Filter for finished status and year >= 2000
    data = data[data["statusId"] == 1]
    data = data.merge(status, on="statusId")
    data = data[data["position"] != "\\N"]
    data["position"] = data["position"].astype(int)
    data = data[data["year"] >= 2000]
    data = data.sort_values(by=["year","round"]).reset_index(drop=True)
    
    # Drop raw points and intermediate cumulative columns to avoid leakage and scale issues
    data = data.drop(columns=[
        "points", "driver_points_before_race", "constructor_points_before_race", "total_points_before_round"
    ])
    
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
    # Calculate previous year's average finishing position for each driver
    driver_year_means = data.groupby(["driverId", "year"])["position"].mean().reset_index()
    driver_year_means.rename(columns={"position": "driver_avg_pos_last_year"}, inplace=True)
    driver_year_means["year"] = driver_year_means["year"] + 1
    data = data.merge(driver_year_means, on=["driverId", "year"], how="left")
    data["driver_avg_pos_last_year"] = data["driver_avg_pos_last_year"].fillna(11.0)

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
    param_dist = {
        'n_estimators': [100, 200, 500],
        'max_depth': [8, 12, 16, None],
        'min_samples_leaf': [2, 5, 10],
        'max_features': ['sqrt', 'log2', None]
    }
    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        base_model, 
        param_distributions=param_dist, 
        n_iter=10, 
        cv=3, 
        scoring='neg_mean_absolute_error', 
        random_state=42,
        n_jobs=-1
    )
    print("Running RandomizedSearchCV hyperparameter tuning...")
    search.fit(x_train, y_train)
    print("Best parameters found:", search.best_params_)
    model = search.best_estimator_
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
    make_plots(y_test, y_pred, "Random Forest")


def xgboost(x_train, y_train, x_test, y_test):
    param_dist = {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'max_depth': [3, 4, 6, 8],
        'min_child_weight': [1, 5, 10],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    base_model = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        base_model, 
        param_distributions=param_dist, 
        n_iter=10, 
        cv=3, 
        scoring='neg_mean_absolute_error', 
        random_state=42,
        n_jobs=-1
    )
    print("Running RandomizedSearchCV hyperparameter tuning for XGBoost...")
    search.fit(x_train, y_train)
    print("Best XGBoost parameters found:", search.best_params_)
    model = search.best_estimator_
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
    make_plots(y_test, y_pred, "XGBoost")

def make_plots(y_test, y_pred, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Determine color scheme based on model name
    color = 'coral' if 'XGBoost' in model_name else 'royalblue'

    # Scatter plot
    axes[0].scatter(y_test, y_pred, alpha=0.5, color=color)
    axes[0].set_xlabel("Actual Position")
    axes[0].set_ylabel("Predicted Position")
    axes[0].set_title(f"Actual vs Predicted Position ({model_name})")

    # Error distribution histogram
    errors = y_test - y_pred
    axes[1].hist(errors, bins=50, color=color, edgecolor='black', alpha=0.8)
    axes[1].set_xlabel("Error")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Error Distribution ({model_name})")

    plt.tight_layout()
    # Save plots to Documentation folder for project reporting
    filename = f"Documentation/scatter_error_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved plot to {filename}")
    plt.show()

    

load_data()
x_train, y_train, x_test, y_test = process_data()
# print("--- RUNNING RANDOM FOREST ---")
# random_forest(x_train, y_train, x_test, y_test)
print("\n--- RUNNING XGBOOST ---")
xgboost(x_train, y_train, x_test, y_test)
print(os.getcwd())
print(y_train.describe())


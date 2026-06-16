import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import os

def load_data():
    races = pd.read_csv("data/raw/races.csv")
    results = pd.read_csv("data/raw/results.csv")
    races = races[["raceId","year","round","circuitId"]]
    results = results[["resultId","raceId","driverId","constructorId","grid","position"]]
    data = results.merge(races, on="raceId")
    data = data[data["position"] != "\\N"]
    data["position"] = data["position"].astype(int)
    data = data.sort_values(by=["year","round"]).reset_index(drop=True)
    data.to_csv("data/processed/cleaned_data.csv", index=False)

def process_data():
    data = pd.read_csv("data/processed/cleaned_data.csv")
    data = temporal_features(data)
    train_data = data[data["year"] < 2022]
    test_data = data[data["year"] >= 2022]
    train_data.to_csv("data/processed/train_data.csv", index=False)
    test_data.to_csv("data/processed/test_data.csv", index=False)
    print("Data processed successfully")
    features_to_drop = ["position", "raceId", "resultId", "round", "year"]

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

    data["driver_avg_grid_last_3"] = (data.groupby("driverId")["grid"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()))
    data["constructor_avg_grid_last_3"] = (data.groupby("constructorId")["grid"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()))

    data["driver_avg_grid_last_5"] = (data.groupby("driverId")["grid"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))
    data["constructor_avg_grid_last_5"] = (data.groupby("constructorId")["grid"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))

    data["driver_avg_pos_last_3"] = data["driver_avg_pos_last_3"].fillna(11)
    data["constructor_avg_pos_last_3"] = data["constructor_avg_pos_last_3"].fillna(11)

    data["driver_avg_pos_last_5"] = data["driver_avg_pos_last_5"].fillna(11)
    data["constructor_avg_pos_last_5"] = data["constructor_avg_pos_last_5"].fillna(11)

    data["driver_avg_grid_last_3"] = data["driver_avg_grid_last_3"].fillna(11)
    data["constructor_avg_grid_last_3"] = data["constructor_avg_grid_last_3"].fillna(11)

    data["driver_avg_grid_last_5"] = data["driver_avg_grid_last_5"].fillna(11)
    data["constructor_avg_grid_last_5"] = data["constructor_avg_grid_last_5"].fillna(11)

    return data

def random_forest(x_train, y_train, x_test, y_test):
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    print("test score:", model.score(x_test, y_test))
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    print("MAE:", mae)
    importance_df = pd.DataFrame({
    "feature": x_train.columns,
    "importance": model.feature_importances_
    })
    importance_df = importance_df.sort_values("importance", ascending=False)
    print(importance_df)

load_data()
x_train, y_train, x_test, y_test = process_data()
random_forest(x_train, y_train, x_test, y_test)
print(os.getcwd())

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import os

def load_data():
    races = pd.read_csv("data/raw/races.csv")
    results = pd.read_csv("data/raw/results.csv")
    races = races[["raceId","year","circuitId"]]
    results = results[["resultId","raceId","driverId","constructorId","grid","position"]]
    data = results.merge(races, on="raceId")
    data = data[data["position"] != "\\N"]
    data["position"] = data["position"].astype(int)
    data.to_csv("data/processed/cleaned_data.csv", index=False)


def random_forest():
    data = pd.read_csv("data/processed/cleaned_data.csv")
    X = data.drop("position", axis=1)
    Y = data["position"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, Y_train)
    print("test score:", model.score(X_test, Y_test))
    Y_pred = model.predict(X_test)
    mae = mean_absolute_error(Y_test, Y_pred)
    print("MAE:", mae)

load_data()
random_forest()
print(os.getcwd())

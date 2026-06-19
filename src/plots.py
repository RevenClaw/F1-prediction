import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the processed training dataset
train_data_path = "data/processed/train_data.csv"
train_df = pd.read_csv(train_data_path)

# 2. Load the raw results and races datasets to filter by year
raw_results_path = "data/raw/results.csv"
raw_df = pd.read_csv(raw_results_path)

races_path = "data/raw/races.csv"
races_df = pd.read_csv(races_path)

# Merge results with races to get the 'year' column
raw_df = raw_df.merge(races_df[["raceId", "year"]], on="raceId")

# Filter raw results to years > 2000
raw_df = raw_df[raw_df["year"] > 2010]

# Clean raw results 'position' column (drop \N and convert to int)
raw_df = raw_df[raw_df["position"] != "\\N"]
raw_df["position"] = raw_df["position"].astype(int)

# Create a side-by-side comparison figure
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Cleaned/Processed Training Dataset Distribution
train_df["position"].value_counts().sort_index().plot(
    kind="bar", 
    color="royalblue", 
    edgecolor="black", 
    alpha=0.85,
    ax=axes[0]
)
axes[0].set_title("Finishing Positions in Cleaned Training Data\n(Only statusId = 1 'Finished')", fontsize=13)
axes[0].set_xlabel("Finishing Position", fontsize=11)
axes[0].set_ylabel("Number of Entries", fontsize=11)
axes[0].grid(axis="y", linestyle="--", alpha=0.7)
axes[0].tick_params(axis='x', rotation=0)

# Plot 2: Raw results.csv Dataset Distribution (Year > 2000)
raw_df["position"].value_counts().sort_index().plot(
    kind="bar", 
    color="darkorange", 
    edgecolor="black", 
    alpha=0.85,
    ax=axes[1]
)
axes[1].set_title("Finishing Positions in Raw results.csv\n(Filtered out '\\N' & Year > 2010)", fontsize=13)
axes[1].set_xlabel("Finishing Position", fontsize=11)
axes[1].set_ylabel("Number of Entries", fontsize=11)
axes[1].grid(axis="y", linestyle="--", alpha=0.7)
axes[1].tick_params(axis='x', rotation=0)

plt.suptitle("F1 Finishing Position Distribution Comparison", fontsize=16, weight="bold")
plt.tight_layout()
plt.show()


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Set the folder containing your CSV files
DATA_FOLDER = r"C:\EV_Research"

station_file = os.path.join(
    DATA_FOLDER, "station_data_dataverse.csv"
)
external_file = os.path.join(
    DATA_FOLDER, "external_sample.csv"
)

# 2. Load the datasets
try:
    station_data = pd.read_csv(station_file)
    external_data = pd.read_csv(external_file)

    print("Datasets loaded successfully!")

except FileNotFoundError as error:
    print("File not found:", error)
    raise SystemExit

# 3. Display dataset information
print("\n--- STATION DATA ---")
print("Rows and columns:", station_data.shape)
print("Column names:", station_data.columns.tolist())
print(station_data.head())

print("\n--- EXTERNAL DATA ---")
print("Rows and columns:", external_data.shape)
print("Column names:", external_data.columns.tolist())
print(external_data.head())

# 4. Check missing values
print("\nMissing values in station data:")
print(station_data.isnull().sum())

print("\nMissing values in external data:")
print(external_data.isnull().sum())

# 5. Descriptive statistics
print("\nDescriptive statistics: Station data")
print(station_data.describe(include="all").T)

print("\nDescriptive statistics: External data")
print(external_data.describe(include="all").T)

# 6. Remove duplicate rows
station_data = station_data.drop_duplicates()
external_data = external_data.drop_duplicates()

print("\nStation data after removing duplicates:",
      station_data.shape)
print("External data after removing duplicates:",
      external_data.shape)

# 7. Plot distributions of numeric variables
numeric_columns = station_data.select_dtypes(
    include=np.number
).columns

if len(numeric_columns) > 0:
    station_data[numeric_columns].hist(
        figsize=(12, 8), bins=20
    )
    plt.suptitle("EV Charging Data Distributions")
    plt.tight_layout()
    plt.show()

# 8. Plot correlations between numeric variables
if len(numeric_columns) >= 2:
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        station_data[numeric_columns].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )
    plt.title("Correlation Between Numeric Variables")
    plt.tight_layout()
    plt.show()

# 9. Save cleaned datasets
station_data.to_csv(
    os.path.join(DATA_FOLDER, "station_data_cleaned.csv"),
    index=False
)

external_data.to_csv(
    os.path.join(DATA_FOLDER, "external_sample_cleaned.csv"),
    index=False
)

print("\nAnalysis completed successfully!")
print("Cleaned datasets have been saved.")

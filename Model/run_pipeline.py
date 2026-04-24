from Model.data_cleaning import build_clean_flights
from Model.feature_engineering import add_feature_engineering


path_2024 = "/Users/crystalguo/Downloads/archive (1)/flight_data_2024.csv"

print("Step 1: start cleaning...", flush=True)
df_clean = build_clean_flights(path_2024)
print("Step 1 done.", flush=True)
print("Cleaned shape:", df_clean.shape, flush=True)

print("Step 2: start feature engineering...", flush=True)
df = add_feature_engineering(df_clean)
print("Step 2 done.", flush=True)

print("Final shape:", df.shape, flush=True)
print(df.head(), flush=True)
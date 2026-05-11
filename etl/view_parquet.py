import pandas as pd

df = pd.read_parquet("processed/season_features.parquet")
print(df.head())

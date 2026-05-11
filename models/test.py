import pandas as pd

# Load the Parquet file
df = pd.read_parquet("processed/season_features.parquet")
print(df.columns)

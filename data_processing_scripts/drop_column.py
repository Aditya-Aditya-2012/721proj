import pandas as pd

# Load your master CSV
df = pd.read_csv("aqi_master_time_series.csv", parse_dates=["datetime"], index_col="datetime")

# Identify all columns for ihbasdilshadgarden
cols_to_drop = [c for c in df.columns if c.startswith("ihbasdilshadgarden_")]

# Drop and save a new CSV
df.drop(columns=cols_to_drop).to_csv("aqi_master_no_ihbas.csv")
print(f"Dropped {len(cols_to_drop)} columns for ihbasdilshadgarden.")

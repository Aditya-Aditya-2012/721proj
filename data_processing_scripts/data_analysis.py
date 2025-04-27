import os
import pandas as pd

# ─── CONFIGURATION ────────────────────────────────────────────────
MASTER_WIDE = "aqi_master_wide.csv"   # ← Update this to your wide CSV filename
OUTPUT_MATRIX = "completeness_matrix.csv"
OUTPUT_POLLUTANT_SUM = "pollutant_completeness.csv"
OUTPUT_STATION_SUM   = "station_completeness.csv"

# 1) Ensure file exists
if not os.path.exists(MASTER_WIDE):
    raise FileNotFoundError(f"Cannot find {MASTER_WIDE}. Please set MASTER_WIDE to your CSV file.")

# 2) Load wide master CSV
df = pd.read_csv(MASTER_WIDE, parse_dates=["datetime"], index_col="datetime")

# 3) Extract station and pollutant names
pairs = [col.split("_", 1) for col in df.columns]
stations = sorted({st for st, _ in pairs})
pollutants = sorted({pol for _, pol in pairs})

# 4) Compute completeness (fraction non-missing) per station-pollutant
matrix = pd.DataFrame(index=stations, columns=pollutants, dtype=float)
for st in stations:
    for pol in pollutants:
        col = f"{st}_{pol}"
        matrix.loc[st, pol] = df[col].notna().mean() if col in df.columns else 0.0

# 5) Summaries
pollutant_summary = matrix.mean(axis=0).reset_index().rename(columns={0: "avg_completeness", "index": "pollutant"})
station_summary   = matrix.mean(axis=1).reset_index().rename(columns={0: "avg_completeness", "index": "station"})

# 6) Save outputs
matrix.to_csv(OUTPUT_MATRIX)
pollutant_summary.to_csv(OUTPUT_POLLUTANT_SUM, index=False)
station_summary.to_csv(OUTPUT_STATION_SUM, index=False)

# 7) Display summaries
# from IPython.display import display
print(f"Completeness matrix saved to {OUTPUT_MATRIX}")
print(f"Pollutant summary saved to {OUTPUT_POLLUTANT_SUM}")
print(f"Station summary saved to {OUTPUT_STATION_SUM}")

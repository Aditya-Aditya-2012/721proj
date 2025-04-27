import pandas as pd

# ─── CONFIGURATION ────────────────────────────────────────────────
MASTER_WIDE       = "aqi_master_wide.csv"
STATION_SUMMARY   = "station_completeness.csv"
FINAL_OUTPUT_CSV  = "aqi_final.csv"
COVERAGE_THRESHOLD = 0.70
KEEP_POLLUTANTS   = {"PM2.5", "PM10", "NO2"}

# ─── 1) Load the wide master DataFrame ─────────────────────────────
df = pd.read_csv(MASTER_WIDE, parse_dates=["datetime"], index_col="datetime")

# ─── 2) Identify well‑covered stations ──────────────────────────────
station_summary = pd.read_csv(STATION_SUMMARY)
good_stations = station_summary.loc[
    station_summary["avg_completeness"] >= COVERAGE_THRESHOLD,
    "station"
].tolist()

# ─── 3) Select only desired pollutants from good stations ────────────
cols_to_keep = [
    col for col in df.columns
    if (col.split("_")[0] in good_stations) and (col.split("_")[1] in KEEP_POLLUTANTS)
]
df_filtered = df[cols_to_keep]

# ─── 4) Time‑based interpolation for short gaps ──────────────────────
df_interp = df_filtered.interpolate(method="time")

# ─── 5) Hour‑of‑day mean fill for any remaining missing values ──────
# Add 'hour' column for grouping
df_interp["hour"] = df_interp.index.hour
# Fill within each hour group
df_interp = (
    df_interp
    .groupby("hour")
    .apply(lambda g: g.fillna(g.mean()))
)
# Drop the helper 'hour' column
df_final = df_interp.drop(columns="hour")

# ─── 6) Save the final DataFrame ────────────────────────────────────
df_final.to_csv(FINAL_OUTPUT_CSV)

# Display the first few rows to verify
print(f"Kept stations (≥ {COVERAGE_THRESHOLD*100:.0f}% coverage): {len(good_stations)}")
print("Example of final interpolated data:")

print(f"\n✅ Final cleaned time series saved to {FINAL_OUTPUT_CSV}")

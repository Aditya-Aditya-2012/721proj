import os
import glob
import pandas as pd

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
BASE_DIR = "/home/civil/btech/ce1210494/721proj/data"   # root containing "Delhi 2018", …, "Delhi 2023"
YEARS    = range(2018, 2024)
FREQ     = "H"                    # hourly resample
OUTPUT   = "aqi_master_wide.csv"

# ─── 1) Load, preprocess, and melt each station file ──────────────────────────
records = []
for year in YEARS:
    folder = os.path.join(BASE_DIR, f"Delhi {year}")
    for fp in glob.glob(os.path.join(folder, "*.xlsx")):
        station = os.path.splitext(os.path.basename(fp))[0]
        
        # Read with header at row 17 (index 16)
        raw = pd.read_excel(fp, header=None)
        header_row = raw[raw.iloc[:,0].astype(str).str.strip()=="From Date"].index[0]
        df = pd.read_excel(fp, header=header_row)
        
        # Parse datetime, drop 'To Date', numeric conversion
        df = df.rename(columns={"From Date":"datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"], dayfirst=True, errors="coerce")
        df = df.drop(columns=["To Date"], errors="ignore").set_index("datetime").sort_index()
        df = df.apply(pd.to_numeric, errors="coerce")
        
        # Resample and interpolate
        df = df.resample(FREQ).mean().interpolate(method="time")
        
        # Melt to long format
        df_long = df.reset_index().melt(
            id_vars=["datetime"],
            var_name="pollutant",
            value_name="value"
        )
        df_long["station"] = station
        records.append(df_long)

# ─── 2) Concatenate all long records ───────────────────────────────────────────
long_df = pd.concat(records, ignore_index=True)

# ─── 3) Pivot to wide format ─────────────────────────────────────────────────
wide = long_df.pivot_table(
    index="datetime",
    columns=["station","pollutant"],
    values="value"
)

# Flatten MultiIndex columns
wide.columns = [f"{st}_{pol}" for st, pol in wide.columns]

# ─── 4) Save to CSV ───────────────────────────────────────────────────────────
wide.to_csv(OUTPUT)
print(f"✅ Wide master time‑series saved to {OUTPUT}")

import os
import glob
import pandas as pd

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
BASE_DIR = "/home/civil/btech/ce1210494/721proj/data"   # Root folder containing "Delhi 2018" ... "Delhi 2023"
YEARS    = range(2018, 2024)
FREQ     = "H"                    # Resample frequency: "H" for hourly
OUTPUT   = "aqi_master_time_series.csv"

all_dfs = []
for year in YEARS:
    folder = os.path.join(BASE_DIR, f"Delhi {year}")
    for fp in glob.glob(os.path.join(folder, "*.xlsx")):
        station = os.path.splitext(os.path.basename(fp))[0]
        
        # 1) Read raw to detect header row
        raw = pd.read_excel(fp, header=None)
        header_idx = raw[raw.iloc[:, 0].astype(str).str.strip() == "From Date"].index
        if header_idx.empty:
            print(f"⚠️  'From Date' header not found in {fp}; skipping")
            continue
        header_row = header_idx[0]
        
        # 2) Read with correct header
        df = pd.read_excel(fp, header=header_row)
        
        # 3) Rename and parse datetime
        df = df.rename(columns={"From Date": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"], dayfirst=True, errors="coerce")
        # Drop rows where datetime failed parsing
        df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
        
        # 4) Drop "To Date" if present
        df = df.drop(columns=["To Date"], errors="ignore")
        
        # 5) Convert pollutant columns to numeric
        df = df.apply(pd.to_numeric, errors="coerce")
        
        # 6) Resample to hourly and average
        df = df.resample(FREQ).mean()
        
        # 7) Interpolate missing values in time
        df = df.interpolate(method="time")
        
        # 8) Prefix columns with station name
        df = df.add_prefix(f"{station}_")
        all_dfs.append(df)

# ─── Combine all stations into master DataFrame ─────────────────────────────────
master = pd.concat(all_dfs, axis=1)

# ─── Forward/backward fill any remaining NaNs at edges ────────────────────────
master = master.ffill().bfill()

# ─── Save to CSV ───────────────────────────────────────────────────────────────
master.to_csv(OUTPUT)
print(f"✅ Master time series written to {OUTPUT}")

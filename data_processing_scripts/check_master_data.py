import pandas as pd

# ─── CONFIG ──────────────────────────────────────────────────────────────
FINAL_CSV  = "aqi_final.csv"           # ← update to your actual filename
REPORT_TXT = "final_data_check.txt"

# ─── 1) Load the final cleaned CSV ─────────────────────────────────────
df = pd.read_csv(
    FINAL_CSV,
    parse_dates=['datetime'],
    index_col='datetime'
)

# ─── 2) Extract station names and pollutants ───────────────────────────
# only consider columns with exactly one underscore separating station_pollutant
valid_cols = [c for c in df.columns if c.count('_') == 1]
stations   = sorted({c.split('_')[0] for c in valid_cols})
pollutants = sorted({c.split('_')[1] for c in valid_cols})

# ─── 3) Check date‑time continuity ────────────────────────────────────
full_idx    = pd.date_range(df.index.min(), df.index.max(), freq='H')
missing_idx = full_idx.difference(df.index)

# ─── 4) Compute missing counts per station and pollutant ──────────────
missing_counts = {
    s: {p: int(df[f"{s}_{p}"].isna().sum()) for p in pollutants if f"{s}_{p}" in df.columns}
    for s in stations
}

# ─── 5) Compute non-missing entry counts per station/pollutant ────────
entry_counts = {
    s: {p: int(df[f"{s}_{p}"].notna().sum()) for p in pollutants if f"{s}_{p}" in df.columns}
    for s in stations
}

# ─── 6) Fully recorded pollutants per station ────────────────────────
fully_recorded = {
    s: [p for p, m in missing_counts[s].items() if m == 0]
    for s in stations
}

# ─── 7) Pollutants fully recorded across all stations ────────────────
common_full = set(pollutants)
for s in stations:
    common_full &= set(fully_recorded[s])

# ─── 8) Write report ─────────────────────────────────────────────────
with open(REPORT_TXT, 'w') as f:
    f.write("FINAL DATASET VALIDATION REPORT\n")
    f.write("="*60 + "\n\n")
    f.write(f"Stations ({len(stations)}): {', '.join(stations)}\n\n")
    f.write(f"Pollutants ({len(pollutants)}): {', '.join(pollutants)}\n\n")
    
    f.write("Date‑Time Continuity:\n")
    if missing_idx.empty:
        f.write(f"  ✔ No missing hourly timestamps from {df.index.min()} to {df.index.max()}\n\n")
    else:
        f.write(f"  ✖ Missing {len(missing_idx)} timestamps:\n")
        for ts in missing_idx[:10]:
            f.write(f"    - {ts}\n")
        if len(missing_idx) > 10:
            f.write("    - ...\n")
        f.write("\n")
    
    f.write("Missing Values per Station & Pollutant:\n")
    for s in stations:
        f.write(f"  Station {s}:\n")
        for p in pollutants:
            key = f"{s}_{p}"
            if key in df.columns:
                f.write(f"    - {p}: {missing_counts[s][p]} missing\n")
        f.write("\n")
    
    f.write("Non‑Missing Entries per Station & Pollutant:\n")
    for s in stations:
        f.write(f"  Station {s}:\n")
        for p in pollutants:
            key = f"{s}_{p}"
            if key in df.columns:
                f.write(f"    - {p}: {entry_counts[s][p]} entries\n")
        f.write("\n")
    
    f.write("Fully Recorded Pollutants per Station:\n")
    for s in stations:
        fr = fully_recorded[s]
        f.write(f"  • {s}: {', '.join(fr) if fr else 'None'}\n")
    f.write("\n")
    
    f.write("Pollutants Fully Recorded Across All Stations:\n")
    f.write("  " + (", ".join(sorted(common_full)) if common_full else "None") + "\n")

print(f"✅ Validation report written to {REPORT_TXT}")

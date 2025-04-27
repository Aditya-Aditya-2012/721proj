import os
import glob

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
BASE_DIR = "/home/civil/btech/ce1210494/721proj/data"  # path to your folders
YEARS    = range(2018, 2024)  # 2018–2023
OUTPUT   = "common_files_report.txt"

# ─── 1) Gather filenames for each year folder ─────────────────────────────────
file_sets = []
for year in YEARS:
    folder = os.path.join(BASE_DIR, f"Delhi {year}")
    files = {
        os.path.basename(fp)
        for fp in glob.glob(os.path.join(folder, "*.xlsx"))
    }
    file_sets.append(files)

# ─── 2) Compute common filenames across all years ───────────────────────────────
common_files = sorted(set.intersection(*file_sets))

# ─── 3) Write report to text file ───────────────────────────────────────────────
with open(OUTPUT, "w") as f:
    f.write(f"Total common files: {len(common_files)}\n")
    f.write("=" * 40 + "\n\n")
    for fname in common_files:
        f.write(f"{fname}\n")

print(f"✅ Report written to {OUTPUT}")

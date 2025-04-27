import pandas as pd

# Paths
excel_path = 'Station Based Details.xlsx'
txt_path   = 'common_files_report.txt'
output_csv = 'station_lat_lon.csv'

# Read common station filenames
with open(txt_path) as f:
    lines = f.read().splitlines()
stations = [line.replace('.xlsx', '') for line in lines if line.endswith('.xlsx')]

# Load and clean coordinates sheet
coords_df = pd.read_excel(excel_path, sheet_name='Station_coordinates')
coords_df.columns = coords_df.columns.str.strip()  # remove extra whitespace


# Mapping from filename key to sheet Station Name
mapping = {
    "alipur": "Alipur",
    "anandvihar": "Anand Vihar",
    "ashokvihar": "Ashok Vihar",
    "ayanagar": "Aya Nagar",
    "bawana": "Bawana",
    "crrimathuraroad": "CRRI Mathura Road",
    "drkarnisinghshootingrange": "Dr. Karni Singh Shooting Range",
    "dtu": "DTU",
    "dwarkasector8": "Dwarka Sec 8",
    "igiairportt3": "IGI Airport",
    "ihbasdilshadgarden": "IHBAS",
    "ito": "ITO",
    "jahangirpuri": "Jahangirpuri",
    "jawaharlalnehrustadium": "Jawaharlal Nehru Stadium",
    "lodhiroadimd": "Lodhi Road IMD",
    "majordhyanchandnationalstadium": "Major Dhyan Chand Stadium",
    "mandirmarg": "Mandir Marg",
    "najafgarh": "Najafgarh",
    "narela": "Narela",
    "nehrunagar": "Nehru Nagar",
    "northcampusdu": "North Campus",
    "nsitdwarka": "NSIT Dwarka",
    "patparganj": "Patparganj",
    "punjabibagh": "Punjabi Bagh",
    "pusadpcc": "Pusa DPCC",
    "pusaimd": "Pusa IMD",
    "rkpuram": "RK Puram",
    "rohini": "Rohini",
    "shadipur": "Shadipur",
    "sirifort": "Sirifort",
    "soniavihar": "Sonia Vihar",
    "sriaurobindomarg": "Sri Aurobindo Marg",
    "vivekvihar": "Vivek Vihar",
    "wazirpur": "Wazirpur"
}

# Build mapping dataframe
rows = []
for key in stations:
    sheet_name = mapping.get(key)
    if sheet_name:
        match = coords_df[coords_df['Station Name'] == sheet_name]
        if not match.empty:
            lat = match.iloc[0]['Latitude']
            lon = match.iloc[0]['Longitude']
        else:
            lat = None
            lon = None
    else:
        lat, lon = None, None
    rows.append({'station': key, 'latitude': lat, 'longitude': lon})

df_final = pd.DataFrame(rows)

# Save and display
df_final.to_csv(output_csv, index=False)

print(f"✅ Saved station lat/lon mapping to {output_csv}")

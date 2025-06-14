import ee
import geemap
import datetime
import sys
import os

# ✅ Initialize Earth Engine
try:
    ee.Initialize(project='ee-i212509')  # Replace with your actual project ID
    print("✅ Earth Engine initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing Earth Engine: {e}")
    sys.exit(1)

# ✅ Define glacier coordinates
GLACIERS = {
    "shisper": ee.Geometry.Point([74.225, 36.110]),
    "passu": ee.Geometry.Point([74.889, 36.467])
}

# ✅ Get user input (glacier name & year)
if len(sys.argv) < 3:
    print("❌ ERROR: Missing arguments. Usage: python glacier.py <glacier_name> <year>")
    sys.exit(1)

glacier_name = sys.argv[1]
year = sys.argv[2]

# ✅ Validate inputs
if glacier_name not in GLACIERS:
    print("❌ ERROR: Invalid glacier name. Choose 'shisper' or 'passu'.")
    sys.exit(1)

if not (2019 <= int(year) <= 2024):
    print("❌ ERROR: Year must be between 2019 and 2024.")
    sys.exit(1)

print(f"🔹 Generating map for {glacier_name} in {year}...")

glacier_point = GLACIERS[glacier_name]

# ✅ Define date range for selected year
start_date = f"{year}-01-01"
end_date = f"{year}-12-31"

# ✅ Load Sentinel-2 Harmonized image collection
sentinel = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
    .filterDate(start_date, end_date) \
    .filterBounds(glacier_point) \
    .sort('CLOUDY_PIXEL_PERCENTAGE') \
    .first()

# ✅ Visualization parameters (True color)
vis_params = {
    'bands': ['B4', 'B3', 'B2'],
    'min': 0,
    'max': 3000,
    'gamma': 1.4
}

# ✅ Create interactive map
Map = geemap.Map(center=[glacier_point.coordinates().get(1).getInfo(), 
                         glacier_point.coordinates().get(0).getInfo()], zoom=12, basemap='SATELLITE')
Map.addLayer(sentinel, vis_params, f'{glacier_name.capitalize()} Glacier {year}')
Map.addLayer(glacier_point, {}, f'{glacier_name.capitalize()} Location')
Map.addLayerControl()

# ✅ Save map as an HTML file
output_dir = "static"
os.makedirs(output_dir, exist_ok=True)  # Ensure the folder exists

map_filename = os.path.join(output_dir, f"{glacier_name}_{year}_map.html")

try:
    Map.save(map_filename)
    print(f"✅ Map saved: {map_filename}")
except Exception as e:
    print(f"❌ ERROR: Failed to save map: {e}")
    sys.exit(1)

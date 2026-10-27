import json
import numpy as np
import rasterio
from pathlib import Path


folder = Path(r"C:\Users\Idktb\Downloads")   # your folder with the .tiff files
TILE = 8

def find(band):
    matches = list(folder.rglob(f"*{band}*.tif*"))
    if not matches:
        raise FileNotFoundError(f"No file containing '{band}' in {folder}")
    return matches[0]

def read_band(band):
    with rasterio.open(find(band)) as src:
        return src.read(1).astype("float32"), src.transform

r, transform = read_band("B04")
g, _ = read_band("B03")
nir, _ = read_band("B08")
try:
    mask, _ = read_band("dataMask")
    valid = mask == 1
except FileNotFoundError:
    valid = np.ones_like(r, dtype=bool)

ndvi = (nir - r) / (nir + r + 1e-6)
vegetation = (ndvi > 0.2) & valid
yellow = (r - g) / (r + g + 1e-6)

def classify(ndvi_mean, ndvi_std, yellow_mean):
    if ndvi_mean > 0.3:
        return "healthy"
    if yellow_mean > 0.08 or ndvi_std > 0.1:
        return "other_stress"
    return "water_stressed"

report, counts, tile_means = [], {"healthy": 0, "other_stress": 0, "water_stressed": 0}, []
height, width = ndvi.shape

for row in range(0, height - TILE + 1, TILE):
    for col in range(0, width - TILE + 1, TILE):
        vt = vegetation[row:row+TILE, col:col+TILE]
        if vt.mean() < 0.2:                       # mostly sand/invalid: skip
            continue
        nt = ndvi[row:row+TILE, col:col+TILE][vt]
        yt = yellow[row:row+TILE, col:col+TILE][vt]
        ndvi_mean, ndvi_std, yellow_mean = nt.mean(), nt.std(), yt.mean()
        tile_means.append(ndvi_mean)

        status = classify(ndvi_mean, ndvi_std, yellow_mean)
        counts[status] += 1
        if status == "healthy":
            continue

        lon, lat = transform * (col + TILE/2, row + TILE/2)
        score = min(max(ndvi_mean / 0.7, 0), 1)
        report.append({"tile": [row, col], "lat": float(lat), "lon": float(lon),
                       "status": status, "health_score": float(score*100),
                       "ndvi_mean": float(ndvi_mean)})

print("tiles assessed:", sum(counts.values()), counts)
print("tile NDVI percentiles:", np.percentile(tile_means, [10, 50, 90]))

with open("output.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"{len(report)} flagged tiles written to output.json")


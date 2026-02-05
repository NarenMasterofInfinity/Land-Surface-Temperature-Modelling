# save as inspect_modis_viirs_layout.py or run in a python shell
from pathlib import Path
import numpy as np
import zarr

PROJECT_ROOT = Path("/home/naren-root/Documents/FYP2/Project")
root = zarr.open_group(str(PROJECT_ROOT / "madurai.zarr"), mode="r")

modis = root["products"]["modis"]["data"]
viirs = root["products"]["viirs"]["data"]

print("MODIS shape:", modis.shape)
print("VIIRS shape:", viirs.shape)

def summarize(arr, name):
    finite = np.isfinite(arr)
    if finite.any():
        print(f"{name}: min={np.nanmin(arr):.3g} max={np.nanmax(arr):.3g} mean={np.nanmean(arr):.3g}")
    else:
        print(f"{name}: all NaN")

# inspect first time step
t = 0
modis_t = modis[t]
viirs_t = viirs[t]

print("\nMODIS bands at t=0:")
for i in range(modis_t.shape[0]):
    summarize(modis_t[i], f"modis band {i}")

print("\nVIIRS bands at t=0:")
for i in range(viirs_t.shape[0]):
    summarize(viirs_t[i], f"viirs band {i}")

# check unique values in mask-like bands
if modis_t.shape[0] >= 6:
    print("\nMODIS band4 unique (valid_day):", np.unique(modis_t[4][np.isfinite(modis_t[4])])[:10])
if viirs_t.shape[0] >= 4:
    print("VIIRS band2 unique (cloud_day):", np.unique(viirs_t[2][np.isfinite(viirs_t[2])])[:10])

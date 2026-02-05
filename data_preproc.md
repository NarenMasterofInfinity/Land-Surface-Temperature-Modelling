# Data Preprocessing Notes

This document captures the preprocessing conventions currently used in this repo. It focuses on:
- origin handling and affine transforms
- canonicalization to a uniform 30 m grid
- no-data handling per source (including Landsat `149`)
- per-source notes (even when no extra processing is done)

## Global Conventions

### Grid metadata and affine transforms
- All ingested products keep their native grid in `madurai.zarr` with CRS and affine transform stored in group attrs (`crs`, `transform`, `width`, `height`, `pixel_size_x`, `pixel_size_y`, `reference_file`).
- `build_dataset_zarr.py` uses the first valid raster of each product as the reference grid and records its CRS and affine transform for that product. If `--warp-to-first` is enabled and a file has a grid mismatch, it is warped to the reference grid.

### Canonicalization to 30 m
- `make_canonical_30m.py` creates `madurai_30m.zarr` by resampling all supported sources onto the Landsat 30 m grid (the canonical grid). The canonical grid is read from `products/landsat` in `madurai.zarr` and written under `grid/` in the output.
- Resampling details:
  - Continuous sources use bilinear resampling.
  - Categorical sources use nearest-neighbor resampling.
  - A `valid` mask is always resampled with nearest neighbor.
- Landsat itself is already on the canonical grid but is still passed through the resampling function for consistency and metadata propagation.

### Valid mask + no-data behavior
- In `madurai.zarr`, each product has:
  - `data` (float32 by default)
  - `valid` mask (`uint8`, 0/1)
- In `build_dataset_zarr.py`:
  - Missing/corrupt files are logged and left as `NaN` in `data` with `valid=0`.
  - For floating-point products, `valid` is computed from `isfinite` data.
  - For integer/categorical products, `valid` is set to 1 everywhere (unless a downstream cleaner overwrites it).
- In `make_canonical_30m.py`, resampling is done on data with invalid pixels masked to `NaN` to prevent NaN bleed across interpolation.

## Origin Handling (ERA5)
- ERA5 uses an explicit override for origin and pixel size to avoid relying on potentially inconsistent file metadata.
- The override grid is defined in `make_canonical_30m.py` with:
  - `ERA5_ORIGIN_X`, `ERA5_ORIGIN_Y`
  - `ERA5_PIXEL_X`, `ERA5_PIXEL_Y`
  - CRS `EPSG:4326`
- This override is applied only when resampling `products/era5` into `products_30m/era5`.

## Canonical 30 m Script Behavior (make_canonical_30m.py)

This section summarizes what the canonicalization script actually does and how it handles grids, transforms, masks, and metadata.

### Canonical grid definition
- Canonical grid is the Landsat 30 m grid read from `products/landsat` in `madurai.zarr`.
- The grid metadata (`crs`, `transform`, `width`, `height`, pixel sizes) is written under `grid/` in `madurai_30m.zarr`.

### What gets resampled
- Resampled to 30 m (Landsat grid): Sentinel‑1, Sentinel‑2, ERA5, DEM, WorldCover, Dynamic World, and Landsat.
- Optional: AlphaEarth is resampled if `--include-alphaearth` is provided.
- Not resampled here: MODIS and VIIRS are intentionally excluded and remain in the raw multi‑grid Zarr.

### Resampling methods
- Continuous sources: bilinear resampling.
- Categorical sources (WorldCover, Dynamic World): nearest‑neighbor resampling.
- Landsat is already on the canonical grid but still passes through the resampler for consistency and metadata propagation.

### Valid mask and NaN handling
- The script requires `data` and `valid` arrays in each source group.
- Before resampling, invalid pixels are masked to `NaN` to prevent interpolation bleed.
- Valid masks are resampled separately using nearest neighbor.
- DEM: if the source valid mask is all zeros, validity is inferred from finite data before resampling.

### ERA5 transform override
- `products/era5` does **not** use its stored transform; instead it uses the explicit ERA5 override grid (origin + pixel size in EPSG:4326).
- This is applied only for ERA5 during resampling.

### Output metadata
- For each output group, the script copies source attrs and adds:
  - `canonical_grid = "landsat_30m"`
  - `source_group`, `source_crs`, `source_transform`
  - `resampling` method
- A build manifest is stored in `zout.attrs["canonical_build_manifest"]`.

## Per-Source Preprocessing Summary

## Source Grid + Resampling Table

This table summarizes CRS handling and resampling in the current pipeline. For most sources, CRS is taken from the product’s own metadata and stored in group attrs; only ERA5 has an explicit override.

| Data Source | CRS (from `gdalinfo`) | Resampling (to 30 m canonical) |
| --- | --- | --- |
| Landsat | EPSG:4326 | Bilinear (identity grid; still resampled for consistency) |
| Sentinel-2 | EPSG:4326 | Bilinear |
| Sentinel-1 | Not found in data folder (no `gdalinfo` sample) | Bilinear |
| ERA5 | EPSG:4326 (override still applied for affine/origin) | Bilinear |
| MODIS | **PROJCRS “MODIS Sinusoidal”** | Not resampled to 30 m (kept native in `madurai.zarr`) |
| VIIRS | **PROJCRS “MODIS Sinusoidal”** | Not resampled to 30 m (kept native in `madurai.zarr`) |
| AlphaEarth | Not found in data folder (no `gdalinfo` sample) | Bilinear (optimized tile/band-batch path optional) |
| DEM | EPSG:4326 | Bilinear |
| WorldCover | EPSG:4326 | Nearest (categorical) |
| Dynamic World | EPSG:4326 | Nearest (categorical) |

### Landsat (LST, 30 m)
- **Canonical grid:** native Landsat grid (used as the canonical reference).
- **Resampling to 30 m:** yes (identity grid, still passed through resampler).
- **No-data handling:**
  - Landsat nodata value `149` is treated as invalid.
  - Temperatures below `273 K` are also treated as invalid for LST QA in analysis scripts.
  - See `check_zarr.py` and `check_landsat_values.py` for these rules.

### Sentinel-2 (monthly)
- **Canonical grid:** resampled to Landsat 30 m.
- **Resampling:** bilinear.
- **No-data handling:** relies on `valid` mask from `build_dataset_zarr.py` (finite values). No source-specific nodata value handled in code yet.

### Sentinel-1 (monthly)
- **Canonical grid:** resampled to Landsat 30 m.
- **Resampling:** bilinear.
- **No-data handling:** relies on `valid` mask (finite values). No source-specific nodata value handled in code yet.

### ERA5 (daily)
- **Canonical grid:** resampled to Landsat 30 m.
- **Resampling:** bilinear.
- **Origin handling:** explicit override for affine transform and origin (see “Origin Handling”).
- **No-data handling:** relies on `valid` mask (finite values). No ERA5-specific nodata constant is applied.

### MODIS (daily, native grid)
- **Canonical grid:** NOT resampled to 30 m (kept native in `madurai.zarr`).
- **No-data handling:**
  - In analysis/cleaning scripts, `-9999` is treated as nodata.
  - LST is stored in Celsius; cleaning scripts mask non-physical values and then convert to Kelvin in analysis (see `check_zarr.py`).
  - `fix_modis_viirs_in_zarr.py` cleans LST bands and writes back to the Zarr.

### VIIRS (daily, native grid)
- **Canonical grid:** NOT resampled to 30 m (kept native in `madurai.zarr`).
- **No-data handling:**
  - LST nodata value `0.0` is treated as invalid.
  - `-9999` is treated as generic nodata for other bands.
  - `fix_modis_viirs_in_zarr.py` cleans LST bands and writes back to the Zarr.

### AlphaEarth (annual)
- **Canonical grid:** resampled to Landsat 30 m (optional).
- **Resampling:** bilinear.
- **Implementation detail:** uses an optimized tile/band-batch warp to reduce RAM in `make_canonical_30m.py`.
- **No-data handling:** relies on `valid` mask (finite values). No explicit source-specific nodata constant applied.

### DEM (static)
- **Canonical grid:** resampled to Landsat 30 m.
- **Resampling:** bilinear.
- **No-data handling:**
  - If DEM `valid` is all zeros, valid is derived from finite data before resampling.

### WorldCover (static, categorical)
- **Canonical grid:** resampled to Landsat 30 m.
- **Resampling:** nearest neighbor.
- **No-data handling:** categorical; no explicit nodata value applied. Valid mask is propagated.

### Dynamic World (static, categorical)
- **Canonical grid:** resampled to Landsat 30 m.
- **Resampling:** nearest neighbor.
- **No-data handling:** categorical; no explicit nodata value applied. Valid mask is propagated.

## Status Gaps / Not Yet Done
- No explicit nodata constants are applied for Sentinel-1, Sentinel-2, ERA5, AlphaEarth, WorldCover, or Dynamic World beyond the valid mask.
- MODIS and VIIRS are not resampled to 30 m in `make_canonical_30m.py`.
- Any additional per-source normalization/scaling should be documented here once implemented.

## Key References
- Canonicalization and resampling: `make_canonical_30m.py`
- Raw Zarr build and grid metadata: `build_dataset_zarr.py`
- Nodata rules and LST cleaning (analysis scripts): `check_zarr.py`, `check_landsat_values.py`, `fix_modis_viirs_in_zarr.py`

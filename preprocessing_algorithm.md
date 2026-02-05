# Preprocessing Algorithm (High Level)

This algorithm describes the preprocessing flow in simple, implementation‑agnostic terms.

## Reference Grids (Sample Files)

| Source | CRS | Origin (x, y) | Pixel Size (x, y) |
| --- | --- | --- | --- |
| Landsat | WGS 84 (EPSG:4326) | (77.98742105389309, 10.316252722828585) | (0.0002694945852359, -0.0002694945852359) |
| Sentinel‑2 | WGS 84 (EPSG:4326) | (77.98742105389309, 10.31607305977176) | (0.0001796630568239, -0.0001796630568239) |
| Sentinel‑1 | Not found (no sample file) | — | — |
| ERA5 | WGS 84 (EPSG:4326) | (77.97376666157446, 10.330625767374498) | (0.0898315284119521, -0.0898315284119521) |
| MODIS | MODIS Sinusoidal | (8539890.644190144, 1147554.6778990142) | (999.9999999995498, -1000.00000000045) |
| VIIRS | MODIS Sinusoidal | (8539890.645999998, 1147554.677000001) | (999.9999999999999, -999.9999999999999) |
| DEM | WGS 84 (EPSG:4326) | (77.98742105389309, 10.316252722828585) | (0.0002694945852359, -0.0002694945852359) |
| WorldCover | WGS 84 (EPSG:4326) | (77.98752577239901, 10.316048146209894) | (8.9831528412e-05, -8.9831528412e-05) |
| Dynamic World | WGS 84 (EPSG:4326) | (77.9875108854215, 10.31607305977176) | (8.9831528412e-05, -8.9831528412e-05) |
| AlphaEarth | Not found (no sample file) | — | — |

1. **Data ingestion**
   1. Read source metadata from a representative raster:
      - CRS
      - Affine transform
      - Width
      - Height

2. **Origin handling and affine transformation**
   1. Use the raster’s affine transform to define the origin (upper‑left corner) and pixel size.
   2. If a source requires a known override, replace the origin/pixel size with the predefined values.
   3. Treat the resulting CRS + affine transform as the authoritative spatial reference for that source.

3. **Canonicalization to a common grid (Landsat grid)**
   1. Define the canonical grid from the Landsat CRS, affine transform, width, and height.
   2. For each source selected for canonicalization, map data from its native grid into the canonical grid.
   3. Produce a canonical valid mask aligned with the Landsat grid.

4. **No‑data handling (bad observations / cloud‑covered observations)**
   1. Identify no‑data pixels using each source’s nodata rules or quality flags.
   2. Update the valid mask (valid=1, invalid=0).
   3. Replace no‑data pixels in the data with `NaN` (or the dataset’s nodata value) before grid mapping.

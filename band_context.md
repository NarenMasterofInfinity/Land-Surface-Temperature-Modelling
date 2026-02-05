UPDATED EXTRA CONTEXT: BAND MEANINGS (ACCURATE TO YOUR DATA)

Dataset: madurai_30m.zarr
Grid: Landsat 30 m canonical grid
Task: High-resolution (30 m) Land Surface Temperature (LST) reconstruction

====================================================
1) Landsat labels (TARGET)
====================================================

labels_30m/landsat/band_01
- Meaning: Landsat-derived Land Surface Temperature (LST), 30 m
- Role: Supervision target (reference LST)
- Physical meaning:
  Surface thermal emission governed by:
  - land cover
  - surface moisture
  - vegetation density
  - urban materials and heat storage
- Usage rule:
  ✔ Target ONLY
  ✘ Never use as input feature

labels_30m/landsat/band_02
- Meaning: Landsat auxiliary / QA / uncertainty band
- Role: Metadata
- Usage rule:
  ✘ Exclude from modeling

====================================================
2) Sentinel-1 SAR features (6 bands)
====================================================
Radar-based structural and moisture information (cloud-independent)

products_30m/sentinel1/band_01
- Meaning: VV backscatter (or equivalent)
- Encodes:
  Urban structure, surface roughness
- LST relevance:
  Strong positive correlation in built-up areas

products_30m/sentinel1/band_02
- Meaning: VH backscatter
- Encodes:
  Vegetation volume scattering
- LST relevance:
  Vegetation lowers surface temperature

products_30m/sentinel1/band_03
- Meaning: VV / VH ratio
- Encodes:
  Built-up vs vegetation separation
- LST relevance:
  Key discriminator of urban heat islands

products_30m/sentinel1/band_04
- Meaning: log(VV)
- Encodes:
  Compressed dynamic range of structural signal
- LST relevance:
  Improves linear and nonlinear separability

products_30m/sentinel1/band_05
- Meaning: log(VH)
- Encodes:
  Moisture and vegetation density
- LST relevance:
  Moist surfaces exhibit lower LST

products_30m/sentinel1/band_06
- Meaning: Texture / composite SAR metric
- Encodes:
  Urban heterogeneity and surface complexity
- LST relevance:
  Captures fine-scale thermal variability in cities

====================================================
3) Sentinel-2 DERIVED SPECTRAL INDICES (13 bands)
====================================================
Optical indices explicitly engineered for surface biophysics and materials.
These are **much stronger than raw bands for linear and tree models**.

----------------------------------------------------
Vegetation indices (cooling effects)
----------------------------------------------------

NDVI
- Normalized Difference Vegetation Index
- Measures: Vegetation density
- LST relevance:
  Strong negative correlation (cooling via evapotranspiration)

EVI
- Enhanced Vegetation Index
- Measures: Dense vegetation with reduced saturation
- LST relevance:
  Better cooling proxy in urban–vegetation mix

SAVI
- Soil-Adjusted Vegetation Index
- Measures: Vegetation in sparse canopy regions
- LST relevance:
  Reduces soil brightness bias

MSAVI
- Modified SAVI
- Measures: Vegetation in very sparse areas
- LST relevance:
  Important in semi-urban / dry regions

----------------------------------------------------
Water & moisture indices (thermal buffering)
----------------------------------------------------

NDWI_water
- Detects open water bodies
- LST relevance:
  Very low surface temperature zones

NDWI_moisture
- Sensitive to vegetation moisture
- LST relevance:
  Moist vegetation → reduced LST

NDMI
- Normalized Difference Moisture Index
- Measures: Canopy water content
- LST relevance:
  Moisture-driven cooling

MNDWI
- Modified NDWI
- Measures: Urban water features
- LST relevance:
  Identifies cooling corridors

----------------------------------------------------
Built-up / impervious surface indices (heating)
----------------------------------------------------

NDBI
- Normalized Difference Built-up Index
- Measures: Impervious urban surfaces
- LST relevance:
  Strong positive correlation (urban heat islands)

UI
- Urban Index
- Measures: Built-up intensity
- LST relevance:
  Captures urban thermal amplification

BSI
- Bare Soil Index
- Measures: Exposed soil
- LST relevance:
  High daytime heating potential

IBI
- Index-Based Built-up Index
- Measures: Composite urban intensity
- LST relevance:
  Very strong urban heat indicator

----------------------------------------------------
Radiative property
----------------------------------------------------

ALBEDO_proxy
- Approximate surface albedo
- Measures: Reflectance vs absorption
- LST relevance:
  Low albedo → higher LST
  High albedo → cooling

====================================================
4) Sentinel-2 CLOUD / OBSERVATION STATISTICS (4 bands)
====================================================

cloud_frac
- Fraction of cloudy observations
- Use:
  ✔ Data reliability indicator
  ✘ Not a physical LST driver

cloud_mask
- Binary cloud presence
- Use:
  ✔ Masking only
  ✘ Never as numeric feature

clear_count
- Number of cloud-free observations
- Use:
  ✔ Confidence / weighting
  ✘ Not direct LST predictor

total_count
- Total observations
- Use:
  ✔ Data completeness indicator
  ✘ Not physical input

IMPORTANT:
- For linear/tree baselines:
  ✘ Exclude all cloud-related bands
- For deep models:
  ✔ May be used as attention / confidence masks

====================================================
5) DEM (terrain controls)
====================================================

static_30m/dem/band_01
- Elevation (m)
- LST relevance:
  Lapse-rate cooling with height

static_30m/dem/band_02
- Slope
- LST relevance:
  Controls solar incidence angle

static_30m/dem/band_03
- Aspect
- LST relevance:
  Directional heating effects

====================================================
6) Land-cover categorical maps
====================================================

static_30m/worldcover/band_01
- ESA WorldCover class label
- Nature: Categorical
- LST relevance:
  Extremely strong semantic cue
- Usage:
  ✘ Exclude from linear/tree
  ✔ Use later via embeddings or one-hot encoding

static_30m/dynamic_world/band_01
- Google Dynamic World land-cover class
- Nature: Categorical (time-aware source)
- Usage:
  ✘ Exclude from linear/tree
  ✔ Use later via soft class probabilities or embeddings



====================================================
7) ERA5 Bands
====================================================

T2M_C

Mean 2 m air temperature

Units: °C

Source: temperature_2m → Kelvin → °C → mean

2. RH2M

Mean 2 m relative humidity

Units: % (0–100)

Derived from temperature_2m + dewpoint_temperature_2m

mean

3. WS10

Mean 10 m wind speed

Units: m/s

Derived as: sqrt(u10² + v10²)

mean

4. TSKIN_C

Mean skin (surface) temperature

Units: °C

Source: skin_temperature → Kelvin → °C → mean

Closest ERA5-Land analogue to LST

5. surface_pressure

Mean surface pressure

Units: Pa

mean

6. total_precip_mm

Total precipitation over the window

Units: mm

Source: total_precipitation (m) × 1000

sum

7. surface_solar_radiation_downwards

Total incoming shortwave radiation

Units: J/m²

sum

8. surface_thermal_radiation_downwards

Total downward longwave radiation

Units: J/m²

sum


====================================================
KEY MODELING IMPLICATIONS
====================================================

1) Sentinel-2 indices already encode physics
- Vegetation cooling
- Built-up heating
- Moisture buffering
→ Linear and tree baselines will be strong

2) SAR complements optical
- Structure and roughness unavailable in optics

3) DEM provides hard physical constraints
- Elevation-driven temperature gradients

4) Cloud bands are NOT predictors
- Only reliability indicators

5) This setup is ideal for:
- Linear → Tree → CNN → Residual → Temporal progression

====================================================
END OF UPDATED BAND CONTEXT
====================================================

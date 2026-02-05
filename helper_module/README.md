# Helper module (Madurai data access)

This folder provides a small helper module for loading the Madurai Zarr datasets
and handling Landsat date alignment. All exports come from `helper_module/helper/utils.py`
(via `helper_module/helper/__init__.py`).

## Quick start

```python
from helper import make_madurai_data, load_subset, describe

md = make_madurai_data()
print(md.vars("madurai_30m"))

ds = load_subset(
    md,
    "madurai_30m",
    start="2019-01-01",
    end="2019-12-31",
)
print(describe(ds))
```

## Core classes

- `DataPaths`: Holds paths to `madurai.zarr`, `madurai_30m.zarr`,
  `madurai_alphaearth_30m.zarr`, and `landsat_dates.json`.
- `LoadSpec`: Structured spec used by `MaduraiData.load()`.
- `MaduraiData`: Main entry point for loading datasets and metadata.

### MaduraiData methods

- `get_dataset(key)`: Open a dataset by key (`"madurai"`, `"madurai_30m"`,
  `"madurai_alphaearth_30m"`).
- `vars(key)`: List variable names. Falls back to Zarr group inspection when
  xarray metadata is missing; names are `group_path/band_name`.
- `time_dim(key)`, `time_values(key)`: Inspect time axis.
- `landsat_dates()`, `landsat_month_pick_map(...)`: Landsat acquisition handling.
- `load(spec)`, `load_many(specs, ...)`: Filtered loading and merging.
- `to_numpy(ds, ...)`: Stack variables into a NumPy tensor.
- `extract_patches(arr, ...)`: Sliding window patch extraction.

## Convenience functions

- `make_madurai_data(chunks="auto", consolidated=None)`: Create a `MaduraiData`
  instance with default project paths.
- `load_all_data(...)`: Merge `madurai.zarr` with optional 30m datasets.
- `load_subset(md, key, ...)`: Load a single dataset with filters.
- `load_landsat_monthly_as_single_date(...)`: Map monthly Landsat time steps
  to a chosen acquisition date per month.
- `describe(ds, max_vars=50)`: Compact summary of a dataset.

## Date utilities

- `parse_date_range`, `month_floor`, `month_ceil_exclusive`,
  `months_in_range`, `pick_landsat_date_for_month`.

## Examples

List variables with fallback Zarr inspection:

```python
from helper import make_madurai_data

md = make_madurai_data()
print(md.vars("madurai_30m"))
```

Load and merge all datasets for a time window:

```python
from helper import make_madurai_data, load_all_data

md = make_madurai_data()
ds = load_all_data(md, start="2019-01-01", end="2020-12-31")
```

Pick Landsat acquisition dates for each month:

```python
from helper import make_madurai_data

md = make_madurai_data()
mp = md.landsat_month_pick_map(
    start="2019-01-01",
    end="2019-12-31",
    strategy="closest_to_mid",
)
print(list(mp.items())[:3])
```

## Dependencies

- Required: `numpy`, `pandas`, `xarray`
- Optional: `zarr` (enables fallback variable listing when xarray metadata is missing)

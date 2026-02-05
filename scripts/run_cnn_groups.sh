#!/usr/bin/env bash
set -euo pipefail

# Group runs for CNN input subsets
python3 baselines/deep/cnn.py --inputs modis,era5 --run-name era5_meteorology_modis
python3 baselines/deep/cnn.py --inputs viirs,era5 --run-name era5_meteorology_viirs
python3 baselines/deep/cnn.py --inputs modis --run-name modis_lst
python3 baselines/deep/cnn.py --inputs viirs --run-name viirs_lst
python3 baselines/deep/cnn.py --inputs modis,s2 --run-name vegetation_indices_modis
python3 baselines/deep/cnn.py --inputs viirs,s2 --run-name vegetation_indices_viirs
python3 baselines/deep/cnn.py --inputs modis,s1,world,dyn --run-name builtup_proxies_modis
python3 baselines/deep/cnn.py --inputs viirs,s1,world,dyn --run-name builtup_proxies_viirs

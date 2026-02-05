#!/usr/bin/env bash
set -euo pipefail

python scripts/make_common_splits.py

# Fusion baselines
python baselines/fusion/starfm.py --lowres viirs --splits metrics/common_date_splits.csv
python baselines/fusion/starfm.py --lowres modis --splits metrics/common_date_splits.csv
python baselines/fusion/ustarfm.py --lowres viirs --splits metrics/common_date_splits.csv
python baselines/fusion/ustarfm.py --lowres modis --splits metrics/common_date_splits.csv
python baselines/fusion/fsdaf.py --lowres viirs --splits metrics/common_date_splits.csv
python baselines/fusion/fsdaf.py --lowres modis --splits metrics/common_date_splits.csv

# Deep baselines
python baselines/deep/resnetmodel.py
python baselines/adv_deep/unet_resnet.py
python baselines/deep/cnn.py
python baselines/deep/convextmodel.py
python baselines/deep/mlpmodel.py
# lightgbm skipped by request

# Linear baselines (update --target if your dataset uses a different name)
python baselines/linear_baselines/linear_baselines.py --dataset madurai_30m --target labels_30m/landsat/data

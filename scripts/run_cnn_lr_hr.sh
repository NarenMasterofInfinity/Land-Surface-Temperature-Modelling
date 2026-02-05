#!/usr/bin/env bash
set -euo pipefail

python3 baselines/deep/cnn_lr_hr/train.py --run-name cnn_lr_hr

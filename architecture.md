# Architecture: BaseNet + ResidualNet (Thermal LR/HR Pipeline)

This document consolidates the full model architecture and training schedule used in the thermal base network + residual network pipeline, including injection/conditioning mechanisms, upsampling head, and training schedules. The details here are derived from the current code in:
- `baselines/deep/cnn_lr_hr/thermal_base/thermal_base_net.py`
- `baselines/deep/cnn_lr_hr/thermal_base/thermal_residual_net.py`
- `baselines/deep/cnn_lr_hr/thermal_base/train.py`

## 1) High-Level Flow (End-to-End)
1. **BaseNet** consumes coarse spatiotemporal inputs (MODIS + VIIRS frames, ERA5, DOY, static context) and predicts a **coarse thermal estimate** (LR).
2. **UpsampleHead** upsamples the BaseNet LR output to HR (tile resolution) to produce `base_hr`.
3. **ResidualNet** consumes `base_hr` along with HR Sentinel-2, Sentinel-1, DEM, landcover, and ERA5 to predict a **residual correction**. The final output is:
   - `y = base_hr + residual`

## 2) BaseNet (ThermalBaseNet)
### 2.1 Inputs
- `modis_frames`: `(B, T=3, 1, H_lr, W_lr)`
- `viirs_frames`: `(B, T=3, 1, H_lr, W_lr)`
- `era5`: `(B, 4, H_lr, W_lr)` (top-4 channels: `ERA5_TOP4 = [1,4,3,0]`)
- `doy`: `(B, 2)` (sin/cos DOY)
- `static`: `(B, 3, H_lr, W_lr)` (DEM + WorldCover + DynamicWorld)
- `modis_masks`, `viirs_masks`: `(B, T=3, 1, H_lr, W_lr)`

### 2.2 Components
**SpatialStem (per frame)**
- `Conv2d(2 -> 16)` + GroupNorm + SiLU
- 2x `BasicBlock` residual blocks
- `ContextBlock` with 3 parallel dilated convs (dilation 1,2,4) + GN + SiLU

**Block Explanations (SpatialStem)**
- **Conv2d(2 → 16) + GroupNorm + SiLU**: 3×3 conv learns 16 spatial feature maps from a 2‑channel input (typically value + mask). GroupNorm stabilizes training with small batches, SiLU provides smooth nonlinearity.
- **BasicBlock (residual)**: `Conv3×3 → GN → SiLU → Conv3×3 → GN → +identity → SiLU`. Keeps spatial size, refines features, and preserves information via skip connection.
- **ContextBlock**: sums three 3×3 convolutions with dilation 1, 2, 4, then GN + SiLU. Expands receptive field without downsampling to capture multi‑scale context.

**TemporalMixer (per sensor)**
- `TinyMamba` (depthwise 1D conv + optional FFN)
- Operates on `(B*H*W, T, C)` sequences; returns last time step

**Sensor Fusion Gate**
- `gate = Conv2d(2 -> 1)` applied to availability masks
- Forces selection when only one sensor is present:
  - If only MODIS present: `g=1`
  - If only VIIRS present: `g=0`
- Fused thermal feature:
  - `f_th = concat(g * f_modis, (1-g) * f_viirs)`

**Meteorological FiLM (conditioning)**
- `film_in = concat(era5, doy, static)`
- `MetFiLM`: `Conv1x1 -> GN -> SiLU -> Conv1x1` produces `(gamma, beta)`
- Conditioning: `f_th = gamma * f_th + beta`

**BaseHead**
- Conv3x3 -> GN -> SiLU -> Dropout
- Conv3x3 -> GN -> SiLU
- Conv1x1 -> 1-channel output (LR thermal)

### 2.3 UpsampleHead
- Projects LR output to channels schedule `[32, 24, 16, 8]`
- For each step: bilinear upsample x2 + 2x Conv3x3 + GN + SiLU
- Final `Conv1x1` to 1 channel

## 3) ResidualNet (Thermal Residual)
### 3.1 Inputs (HR)
- `base_hr`: `(B, 1, H_hr, W_hr)`
- `s2`: `(B, s2_ch, H_hr, W_hr)`
- `s1`: `(B, s1_ch, H_hr, W_hr)` (optional)
- `dem`: `(B, 1, H_hr, W_hr)` (optional)
- `lc`: `(B, 1, H_hr, W_hr)` landcover categorical
- `era5`: `(B, 4, H_hr, W_hr)`

### 3.2 Adapters (Input Injection)
Each adapter uses **availability masking** by concatenating input with finite-mask and gating the output.
- **S2 Adapter**: `ConvBlock + 3x BasicBlock + AvailabilityGate`
- **S1 Adapter**: `ConvBlock + 1x BasicBlock + AvailabilityGate`
- **DEM Adapter**: `ConvBlock + 1x ConvBlock + AvailabilityGate`
- **LC Adapter**:
  - Categorical: `Embedding -> 1x1 Conv -> GN -> SiLU`
  - One-hot supported (not used in current config)

### 3.3 Residual U-Net Trunk
- `ResidualUNet(in_ch, channels=[64,96,128,192])`
- Encoder: 3 downsample stages
- Decoder: bilinear upsample + concat skip + ConvBlocks
- Outputs multi-scale features `(d1, d2, d3, e4)`

### 3.4 FiLM Injection of Base + ERA5
At each scale, FiLM conditions with `concat(base_hr, era5)`:
- `film_i = FiLM(base_ch + era5_ch -> feat_ch_i)`
- `d_i = gamma_i * d_i + beta_i`

### 3.5 Residual Head
- Conv3x3 -> GN -> SiLU -> Dropout2d
- Conv1x1 -> residual prediction `r`
- Optional `tanh` clamp: `r = tanh(r) * residual_tanh` (default `15.0`)
- Final output: `y = base_hr + r`

## 4) Training Schedule
### 4.1 Base Stage
| Setting | Value |
| --- | --- |
| Epochs | `1..80` |
| Optimizer | AdamW |
| Param groups | `main params: lr=peak_lr, wd=1e-4`; `mamba params: lr=0.5*peak_lr, wd=0.0` |
| Peak LR | `2e-4` if batch size <= 4; `3e-4` otherwise |
| LR schedule | Cosine with warmup (per-step) |
| Total steps | `80 * len(train_loader)` |
| Warmup steps | `5 * len(train_loader)` |
| Warmup start | `1e-5` |
| End LR | `3e-6` |
| Loss | Smooth L1 on valid pixels of normalized target |
| Gradient clipping | `clip_grad_norm_(base_net, 1.0)` |
| Early stopping | patience = 12 (val RMSE) |

### 4.2 Residual Stage
The residual stage is trained in two phases, both using **cosine LR with warmup**.

**Phase R0**
| Setting | Value |
| --- | --- |
| Epochs | `1..90` |
| Trainable modules | ResidualNet only (BaseNet + UpsampleHead frozen) |
| Optimizer | AdamW(res_net) |
| LR / WD | `lr=2e-4`, `wd=5e-4` |
| Warmup start | `1e-5` |
| End LR | `2e-6` |

**Phase R1**
| Setting | Value |
| --- | --- |
| Epochs | `91..120` |
| Trainable modules | ResidualNet + BaseNet + UpsampleHead |
| Optimizer | AdamW with two param groups |
| Param group 1 | `res_net: lr=5e-5, wd=5e-4` |
| Param group 2 | `base + up_head: lr=0.05*lr_peak (2.5e-6), wd=1e-4` |
| Warmup start | `5e-6` |
| End LR | `5e-6` |

**Common Residual Stage Details**
| Setting | Value |
| --- | --- |
| Warmup steps | `5 * len(train_loader)` |
| Total steps | `(end_epoch - start_epoch + 1) * len(train_loader)` |
| Loss | Smooth L1 on valid HR pixels |
| Gradient clipping | `clip_grad_norm_(res + base + up_head, 1.0)` when base is trainable |
| Early stopping | patience = 15 (val RMSE) |

## 5) Torchinfo Summaries (Pending)
Torchinfo is not available in the current environment (`python3` lacks `pip`/`torchinfo`). To generate exact parameter counts and layer-by-layer summaries, run the following after installing `torchinfo`:

```bash
python3 -m pip install --user torchinfo
python3 scripts/print_arch_summaries.py
```

The `scripts/print_arch_summaries.py` script instantiates `ThermalBaseNet`, `UpsampleHead`, and `ResidualNet` using shapes inferred from the local zarr datasets (if present) and prints the summaries to stdout for capture.

## 6) Injection/Conditioning Summary (Checklist)
- **Mask-based gating**: applied for MODIS/VIIRS availability (BaseNet) and S1/S2/DEM (ResidualNet)
- **Temporal injection**: TinyMamba sequence mixing in BaseNet
- **Sensor fusion gate**: learned gating between MODIS/VIIRS
- **FiLM conditioning**:
  - BaseNet: `era5 + doy + static` -> `(gamma, beta)`
  - ResidualNet: `base_hr + era5` -> `(gamma, beta)` at 4 UNet scales
- **Residual clamp**: `tanh` scaled to 15°C

---

If you want, I can add the torchinfo summaries directly once `torchinfo` is installable in this environment.

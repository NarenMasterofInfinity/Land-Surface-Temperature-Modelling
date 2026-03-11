param(
  [string]$Config = "basenet_1km/config.yaml",
  [switch]$Smoke = $false
)

$ErrorActionPreference = "Stop"

if ($Smoke) {
  python basenet_1km/train_basenet.py --config $Config --smoke
} else {
  python basenet_1km/train_basenet.py --config $Config
}


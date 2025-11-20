#!/usr/bin/env bash
# run_all.sh - environment + split creation + placeholder commands
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
echo "ROOT: $ROOT"

# create venv (if not exists)
python3 -m venv "${ROOT}/venv" || true
source "${ROOT}/venv/bin/activate"
pip install --upgrade pip

# install minimal deps
pip install numpy scipy scikit-learn pyyaml matplotlib torch torchvision

# create deterministic splits for ADSB and WiFi (example)
mkdir -p "${ROOT}/splits"
python3 "${ROOT}/splits.py" --metadata "${ROOT}/data/adsb/metadata.csv" --protocol ADSB --outdir "${ROOT}/splits" --seed 42 --time_stratify
python3 "${ROOT}/splits.py" --metadata "${ROOT}/data/wifi/metadata.csv" --protocol WiFi --outdir "${ROOT}/splits" --seed 42 --time_stratify

echo "Splits generated in ${ROOT}/splits/"
echo "Next steps (examples):"
echo "  - Implement train.py that uses config.yaml + splits json, NIT (nit.py), ARR (arr.py) and trains CRL-SEI."
echo "  - After training, extract embeddings and distances -> fit weibull per class using weibull_calibrator.py"
echo ""
echo "Example (pseudo):"
echo "  python train.py --config config.yaml --splits splits/ADSB_splits_seed42.json"
echo "  python extract_embeddings.py --model checkpoints/best.pth --out emb_val.npy"
echo "  python fit_weibull_posthoc.py --emb emb_val.npy --labels val_labels.npy --tail_size 20"
echo "Done."

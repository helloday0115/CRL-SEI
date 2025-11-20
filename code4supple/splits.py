"""splits.py
Utilities to create deterministic train/val/test splits for ADS-B and WiFi datasets.

Expected input:
  - metadata CSV with at least these columns: sample_id, emitter_id, timestamp (ISO or numeric)
  - protocol: 'ADSB' or 'WiFi'

Purpose:
  - produce reproducible per-protocol splits (train/val/test) and optionally emitter-level splits
  - supports time-stratified ordering to respect temporal shifts (useful for domain generalization)

Example:
  python splits.py --metadata data/adsb/metadata.csv --protocol ADSB --outdir ./splits --seed 42 --train_frac 0.7 --val_frac 0.15 --test_frac 0.15 --time_stratify
"""
import argparse, json, random, os
from collections import defaultdict

def make_splits(metadata_csv, protocol, seed=42, train_frac=0.7, val_frac=0.15, test_frac=0.15, time_stratify=True):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "fractions must sum to 1"
    rows = []
    with open(metadata_csv, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split(',')
        for line in f:
            vals = line.strip().split(',')
            if len(vals) != len(header):
                continue
            row = dict(zip(header, vals))
            rows.append(row)
    # Optional time stratification: sort by timestamp if present
    if time_stratify and 'timestamp' in header:
        rows = sorted(rows, key=lambda r: r.get('timestamp','0'))
    # group by emitter (so we can optionally split emitters across sets)
    emitters = defaultdict(list)
    for r in rows:
        emitters[r['emitter_id']].append(r)
    # deterministic emitter ordering + shuffle with seed for randomness but reproducible
    emitter_ids = sorted(list(emitters.keys()))
    rnd = random.Random(seed)
    rnd.shuffle(emitter_ids)
    n = len(emitter_ids)
    n_train = max(1, int(round(train_frac * n)))
    n_val = max(1, int(round(val_frac * n)))
    # ensure at least 1 test
    n_test = max(1, n - n_train - n_val)
    train_e = set(emitter_ids[:n_train])
    val_e = set(emitter_ids[n_train:n_train+n_val])
    test_e = set(emitter_ids[n_train+n_val: n_train+n_val+n_test])
    # collect sample ids
    train, val, test = [], [], []
    for eid, items in emitters.items():
        ids = [it['sample_id'] for it in items]
        if eid in train_e:
            train.extend(ids)
        elif eid in val_e:
            val.extend(ids)
        else:
            test.extend(ids)
    out = {
        'protocol': protocol,
        'seed': seed,
        'train': train,
        'val': val,
        'test': test,
        'train_emitters': list(train_e),
        'val_emitters': list(val_e),
        'test_emitters': list(test_e)
    }
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--protocol', choices=['ADSB','WiFi'], required=True)
    parser.add_argument('--outdir', default='./splits')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_frac', type=float, default=0.7)
    parser.add_argument('--val_frac', type=float, default=0.15)
    parser.add_argument('--test_frac', type=float, default=0.15)
    parser.add_argument('--time_stratify', action='store_true', help='Sort by timestamp before splitting')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    result = make_splits(args.metadata, args.protocol, seed=args.seed, train_frac=args.train_frac, val_frac=args.val_frac, test_frac=args.test_frac, time_stratify=args.time_stratify)
    outpath = os.path.join(args.outdir, f"{args.protocol}_splits_seed{args.seed}.json")
    with open(outpath, 'w') as f:
        json.dump(result, f, indent=2)
    print('Wrote', outpath)

if __name__ == '__main__':
    main()

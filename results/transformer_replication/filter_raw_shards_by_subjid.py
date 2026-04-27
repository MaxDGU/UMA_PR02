"""Concatenate raw whole-number trace shards filtered to a chosen set of subjids.

Reads all shards matching --shard-glob, keeps only rows whose `subjid` is in
the subjid-include list, and writes a single CSV (or .csv.gz) suitable for
feeding into the existing translator.
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--shard-glob', required=True,
                   help='Glob for raw shards, e.g. ".../uma_traces_exhaustive_whole_*.csv"')
    p.add_argument('--subjids-file', required=True,
                   help='Text file with one subjid per line')
    p.add_argument('--out-csv', required=True,
                   help='Output CSV (use .csv.gz for compressed)')
    p.add_argument('--chunksize', type=int, default=200000)
    return p.parse_args()


def main():
    a = parse_args()
    with open(a.subjids_file) as f:
        subjids = {int(s.strip()) for s in f if s.strip()}
    print(f"Filtering to {len(subjids)} subjids")

    shards = sorted(glob.glob(a.shard_glob))
    print(f"Found {len(shards)} shards")

    os.makedirs(os.path.dirname(a.out_csv) or '.', exist_ok=True)
    total_rows_in = 0
    total_rows_out = 0
    first_chunk = True
    compression = 'gzip' if a.out_csv.endswith('.gz') else None

    for i, shard in enumerate(shards):
        # only read shards whose subjid range potentially overlaps our subset
        # (shards are named ..._<lo>_<hi>.csv with hi exclusive in our generation)
        for chunk in pd.read_csv(shard, chunksize=a.chunksize):
            total_rows_in += len(chunk)
            kept = chunk[chunk['subjid'].isin(subjids)]
            total_rows_out += len(kept)
            if not kept.empty:
                kept.to_csv(a.out_csv, mode='w' if first_chunk else 'a',
                            header=first_chunk, index=False,
                            compression=compression)
                first_chunk = False
        if (i + 1) % 5 == 0 or i + 1 == len(shards):
            print(f"  shard {i+1}/{len(shards)}: in={total_rows_in:,} out={total_rows_out:,}")

    print(f"\nKept {total_rows_out:,} of {total_rows_in:,} rows "
          f"({100*total_rows_out/max(total_rows_in,1):.2f}%)")
    print(f"Wrote -> {a.out_csv}")


if __name__ == '__main__':
    main()

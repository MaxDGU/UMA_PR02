#!/usr/bin/env python3
"""Shard packaged translated synthetic algebra CSVs into Parquet files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_DATASET_DIR = Path(
    "cognitive_tutor/data/processed/algebra_2006_2007/"
    "translated_synthetic_algebra_v3_human_mixed"
)
DEFAULT_PREFIX = "singlevar_algebra_synth_v3_human_mixed"
DEFAULT_OUT_DIR = DEFAULT_DATASET_DIR / "parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rows-per-shard", type=int, default=50_000)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return parser.parse_args()


def shard_split(
    input_csv: Path,
    out_split_dir: Path,
    rows_per_shard: int,
    compression: str,
) -> dict[str, object]:
    if rows_per_shard <= 0:
        raise ValueError("rows_per_shard must be positive")
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing split CSV: {input_csv}")
    out_split_dir.mkdir(parents=True, exist_ok=True)
    for old in out_split_dir.glob("part-*.parquet"):
        old.unlink()

    shards: list[dict[str, object]] = []
    total_rows = 0
    columns: list[str] | None = None
    for shard_idx, chunk in enumerate(pd.read_csv(input_csv, chunksize=rows_per_shard, dtype=str, keep_default_na=False)):
        if columns is None:
            columns = list(chunk.columns)
        shard_path = out_split_dir / f"part-{shard_idx:05d}.parquet"
        chunk.to_parquet(shard_path, index=False, engine="pyarrow", compression=compression)
        rows = int(len(chunk))
        total_rows += rows
        shards.append(
            {
                "path": str(shard_path),
                "rows": rows,
                "bytes": shard_path.stat().st_size,
            }
        )
    return {
        "input_csv": str(input_csv),
        "out_dir": str(out_split_dir),
        "rows": total_rows,
        "num_shards": len(shards),
        "columns": columns or [],
        "shards": shards,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def shard_dataset(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    prefix: str = DEFAULT_PREFIX,
    out_dir: Path = DEFAULT_OUT_DIR,
    rows_per_shard: int = 50_000,
    compression: str = "zstd",
    splits: list[str] | None = None,
) -> dict[str, object]:
    split_names = splits or ["train", "val", "test"]
    out_dir.mkdir(parents=True, exist_ok=True)
    split_summaries = {}
    for split in split_names:
        input_csv = dataset_dir / f"{prefix}_{split}.csv.gz"
        split_summaries[split] = shard_split(
            input_csv=input_csv,
            out_split_dir=out_dir / split,
            rows_per_shard=rows_per_shard,
            compression=compression,
        )

    manifest = {
        "dataset_dir": str(dataset_dir),
        "prefix": prefix,
        "out_dir": str(out_dir),
        "rows_per_shard": rows_per_shard,
        "compression": compression,
        "splits": split_summaries,
        "prompt_col": "instruction_nl",
        "response_col": "response_nl",
        "problem_col": "prob",
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = shard_dataset(
        dataset_dir=args.dataset_dir,
        prefix=args.prefix,
        out_dir=args.out_dir,
        rows_per_shard=args.rows_per_shard,
        compression=args.compression,
        splits=args.splits,
    )
    print(f"out_dir={manifest['out_dir']}", flush=True)
    for split, summary in manifest["splits"].items():
        print(
            f"{split}: rows={summary['rows']} shards={summary['num_shards']}",
            flush=True,
        )
    print(f"manifest_json={Path(manifest['out_dir']) / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()

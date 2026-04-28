import csv
import gzip
from pathlib import Path

import pandas as pd

from cognitive_tutor.shard_translated_synthetic_algebra_parquet import shard_dataset


COLS = ["instruction_nl", "response_nl", "prob", "correct"]


def write_split(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLS)
        writer.writeheader()
        for idx in range(rows):
            writer.writerow(
                {
                    "instruction_nl": f"Solve {idx}",
                    "response_nl": f"work\n### answer: {idx}",
                    "prob": f"x={idx}",
                    "correct": str(idx % 2),
                }
            )


def test_shard_dataset_to_parquet(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    for split, rows in [("train", 5), ("val", 2), ("test", 1)]:
        write_split(dataset_dir / f"toy_{split}.csv.gz", rows)

    out_dir = tmp_path / "parquet"
    manifest = shard_dataset(
        dataset_dir=dataset_dir,
        prefix="toy",
        out_dir=out_dir,
        rows_per_shard=2,
        compression="zstd",
    )

    assert manifest["splits"]["train"]["num_shards"] == 3
    assert manifest["splits"]["train"]["rows"] == 5
    first = pd.read_parquet(out_dir / "train" / "part-00000.parquet")
    assert first.columns.tolist() == COLS
    assert len(first) == 2
    assert (out_dir / "manifest.json").exists()

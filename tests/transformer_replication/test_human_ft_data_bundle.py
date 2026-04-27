from pathlib import Path
import hashlib
import json

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "human_ft"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_human_ft_data_bundle_row_counts() -> None:
    raw = pd.read_csv(DATA_DIR / "data.csv", encoding="cp1252")
    train_nlp = pd.read_csv(DATA_DIR / "data_train_nlp.csv")
    val_nlp = pd.read_csv(DATA_DIR / "data_val_nlp.csv")
    trainval_nlp = pd.read_csv(DATA_DIR / "all_data" / "data_trainval_nlp.csv")

    assert len(raw) == 384
    assert raw["subjid"].nunique() == 48
    assert raw["prob"].nunique() == 8

    assert len(train_nlp) == 288
    assert train_nlp["subjid"].nunique() == 36
    assert train_nlp["prob"].nunique() == 8

    assert len(val_nlp) == 96
    assert val_nlp["subjid"].nunique() == 12
    assert val_nlp["prob"].nunique() == 8

    assert len(trainval_nlp) == 384


def test_human_ft_manifest_matches_bundle_and_uses_relative_paths() -> None:
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["summary"]["raw_corpus"]["rows"] == 384
    assert manifest["summary"]["train_nlp"]["rows"] == 288
    assert manifest["summary"]["val_nlp"]["rows"] == 96
    assert manifest["summary"]["trainval_nlp"]["rows"] == 384
    assert manifest["summary"]["correct_only"] == {"train_rows": 147, "val_rows": 29}

    payload = json.dumps(manifest)
    assert "/scratch/" not in payload
    assert "/n/fs/" not in payload
    assert len(manifest["files"]) == 15
    for entry in manifest["files"]:
        path = REPO_ROOT / entry["path"]
        assert path.exists()
        assert path.stat().st_size == entry["size_bytes"]
        assert _sha256(path) == entry["sha256"]
        assert not Path(entry["source_path"]).is_absolute()

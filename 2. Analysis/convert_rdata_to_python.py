#!/usr/bin/env python3
"""Convert RData/RDS files to Python-friendly artifacts.

Examples:
  python '2. Analysis/convert_rdata_to_python.py'
  python '2. Analysis/convert_rdata_to_python.py' '2. Analysis/sim ALL29_BCD data.Rdata' --output-dir converted
"""

from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert .RData/.Rdata/.rds files to .npy/.npz and tabular files that are easy to use in Python."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Input R files. If omitted, the script scans the current directory recursively.",
    )
    parser.add_argument(
        "--output-dir",
        default="converted_rdata",
        help="Directory where converted files are written (default: converted_rdata).",
    )
    parser.add_argument(
        "--skip-csv",
        action="store_true",
        help="Skip CSV export for pandas DataFrames.",
    )
    parser.add_argument(
        "--skip-npz",
        action="store_true",
        help="Skip NPZ export for pandas DataFrames.",
    )
    parser.add_argument(
        "--as-fractions",
        action="store_true",
        help=(
            "Convert arithmetic decimal values/expressions in DataFrames to fraction strings "
            "(e.g., 0.41 -> 41/100)."
        ),
    )
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("_")
    return cleaned or "object"


def discover_inputs(inputs: Iterable[str]) -> List[Path]:
    if inputs:
        paths = [Path(p).expanduser().resolve() for p in inputs]
    else:
        cwd = Path.cwd()
        discovered: List[Path] = []
        for p in cwd.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".rdata", ".rds"}:
                discovered.append(p.resolve())
        paths = sorted(discovered)

    files = [p for p in paths if p.is_file() and p.suffix.lower() in {".rdata", ".rds"}]
    return sorted(files)


def normalize_loaded_objects(loaded: Dict[object, object], default_name: str) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in loaded.items():
        if key in (None, "", "None"):
            key = default_name
        out[sanitize_name(str(key))] = value
    return out


def load_with_pyreadr(path: Path) -> Dict[str, object]:
    try:
        import pyreadr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyreadr is required for primary loading. Install it with: python -m pip install pyreadr"
        ) from exc

    loaded = pyreadr.read_r(str(path))
    default_name = sanitize_name(path.stem)
    out = normalize_loaded_objects(loaded, default_name)
    if not out:
        out[default_name] = None
    return out


def load_with_rdata(path: Path) -> Dict[str, object]:
    try:
        import rdata
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "rdata is required for fallback loading. Install it with: python -m pip install rdata"
        ) from exc

    parsed = rdata.parser.parse_file(str(path))
    converted = rdata.conversion.convert(parsed)
    default_name = sanitize_name(path.stem)

    if isinstance(converted, dict):
        out = normalize_loaded_objects(converted, default_name)
        if not out:
            out[default_name] = None
        return out

    return {default_name: converted}


def load_r_objects(path: Path) -> Dict[str, object]:
    errors: List[str] = []
    try:
        return load_with_pyreadr(path)
    except Exception as exc:
        errors.append(f"pyreadr failed ({exc})")

    try:
        return load_with_rdata(path)
    except Exception as exc:
        errors.append(f"rdata failed ({exc})")

    raise RuntimeError("; ".join(errors))


def object_base_path(output_root: Path, input_file: Path, object_name: str) -> Path:
    cwd = Path.cwd().resolve()
    try:
        relative_parent = input_file.resolve().parent.relative_to(cwd)
    except ValueError:
        relative_parent = input_file.resolve().parent

    target_dir = output_root / relative_parent / sanitize_name(input_file.stem)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / sanitize_name(object_name)


def with_extension(base: Path, ext: str) -> Path:
    return base.parent / f"{base.name}{ext}"


ARITH_NUMERIC_COLUMNS = {
    "op1",
    "op2",
    "key",
    "key_val",
    "resp",
    "resp_val",
    "ansval",
    "opt1",
    "opt2",
}

ARITH_TEXT_COLUMNS = {
    "prob",
    "opt1",
    "opt2",
    "resp",
    "work_operand_1",
    "work_operand_2",
    "work_answer",
}

DECIMAL_TOKEN_RE = re.compile(r"(?<![0-9A-Za-z_./-])-?(?:\d+\.\d+|\.\d+|\d+\.?)(?:[eE][+-]?\d+)?")


def number_to_fraction_text(value: object) -> object:
    if pd.isna(value):
        return value
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return value

    text = str(value).strip()
    if text == "":
        return value

    try:
        dec = Decimal(text)
    except (InvalidOperation, ValueError):
        return value

    if not dec.is_finite():
        return value

    frac = Fraction(dec)
    if frac.denominator == 1:
        return str(frac.numerator)
    return f"{frac.numerator}/{frac.denominator}"


def expression_decimals_to_fractions(text: object) -> object:
    if pd.isna(text):
        return text

    raw = str(text)
    if "." not in raw and "e" not in raw.lower():
        return text

    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        # Leave integers untouched; convert decimal/scientific literals.
        if "." not in token and "e" not in token.lower():
            return token
        converted = number_to_fraction_text(token)
        return str(converted)

    return DECIMAL_TOKEN_RE.sub(repl, raw)


def fractionize_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    changed: List[str] = []

    for col in out.columns:
        name = str(col)
        series = out[col]

        if name in ARITH_NUMERIC_COLUMNS and pd.api.types.is_numeric_dtype(series):
            out[col] = series.map(number_to_fraction_text)
            changed.append(name)
            continue

        if name in ARITH_TEXT_COLUMNS:
            out[col] = series.map(expression_decimals_to_fractions)
            changed.append(name)

    return out, sorted(set(changed))


def save_dataframe(df: pd.DataFrame, base: Path, skip_csv: bool, skip_npz: bool) -> List[Path]:
    written: List[Path] = []

    if not skip_csv:
        csv_path = with_extension(base, ".csv")
        df.to_csv(csv_path, index=True)
        written.append(csv_path)

    if not skip_npz:
        npz_path = with_extension(base, ".npz")
        np.savez_compressed(
            npz_path,
            data=df.to_numpy(),
            columns=df.columns.astype(str).to_numpy(),
            index=df.index.astype(str).to_numpy(),
        )
        written.append(npz_path)

    return written


def save_series(series: pd.Series, base: Path) -> List[Path]:
    written: List[Path] = []

    values_path = with_extension(base, ".npy")
    np.save(values_path, series.to_numpy(), allow_pickle=True)
    written.append(values_path)

    index_path = base.parent / f"{base.name}_index.npy"
    np.save(index_path, series.index.astype(str).to_numpy(), allow_pickle=True)
    written.append(index_path)

    return written


def save_array(arr: np.ndarray, base: Path) -> List[Path]:
    path = with_extension(base, ".npy")
    np.save(path, arr, allow_pickle=True)
    return [path]


def save_fallback(obj: object, base: Path) -> List[Path]:
    path = with_extension(base, ".npy")
    np.save(path, np.array(obj, dtype=object), allow_pickle=True)
    return [path]


def write_metadata(base: Path, metadata: Dict[str, object]) -> Path:
    meta_path = with_extension(base, ".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return meta_path


def convert_object(
    obj: object,
    base: Path,
    skip_csv: bool,
    skip_npz: bool,
    as_fractions: bool,
) -> Tuple[List[Path], Dict[str, object]]:
    metadata: Dict[str, object] = {
        "python_type": type(obj).__name__ if obj is not None else "NoneType",
    }

    if isinstance(obj, pd.DataFrame):
        if as_fractions:
            obj, changed_cols = fractionize_dataframe(obj)
            if changed_cols:
                metadata["fractionized_columns"] = changed_cols
        metadata["shape"] = list(obj.shape)
        metadata["columns"] = [str(c) for c in obj.columns]
        files = save_dataframe(obj, base, skip_csv=skip_csv, skip_npz=skip_npz)
        return files, metadata

    if isinstance(obj, pd.Series):
        metadata["shape"] = [int(obj.shape[0])]
        metadata["name"] = str(obj.name)
        files = save_series(obj, base)
        return files, metadata

    if isinstance(obj, np.ndarray):
        metadata["shape"] = list(obj.shape)
        metadata["dtype"] = str(obj.dtype)
        files = save_array(obj, base)
        return files, metadata

    files = save_fallback(obj, base)
    return files, metadata


def main() -> int:
    args = parse_args()
    input_files = discover_inputs(args.inputs)
    if not input_files:
        print("No .RData/.Rdata/.rds files found.")
        return 1

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    total_objects = 0
    total_artifacts = 0

    for path in input_files:
        print(f"[load] {path}")
        try:
            objects = load_r_objects(path)
        except Exception as exc:
            print(f"  [error] failed to read {path}: {exc}")
            continue

        for object_name, obj in objects.items():
            base = object_base_path(output_root, path, object_name)
            artifacts, metadata = convert_object(
                obj,
                base,
                skip_csv=args.skip_csv,
                skip_npz=args.skip_npz,
                as_fractions=args.as_fractions,
            )
            artifacts.append(write_metadata(base, metadata))
            total_objects += 1
            total_artifacts += len(artifacts)
            rel_artifacts = [str(p.relative_to(output_root)) for p in artifacts]
            print(f"  [ok] {object_name} -> {', '.join(rel_artifacts)}")

    print(f"Converted {total_objects} object(s) into {total_artifacts} artifact(s) at {output_root}")
    return 0 if total_objects else 2


if __name__ == "__main__":
    raise SystemExit(main())

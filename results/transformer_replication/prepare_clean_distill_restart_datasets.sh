#!/bin/bash
set -euo pipefail

if [[ -n "${REPO_ROOT:-}" ]]; then
  ROOT_DIR="${REPO_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(pwd)"
fi
cd "${ROOT_DIR}"

BASE_DIR="${ROOT_DIR}/results/transformer_replication"
TRANSLATOR="${BASE_DIR}/translate_uma_traces_to_nlp.py"

INPUT_CSV="${INPUT_CSV:-${BASE_DIR}/synth_unique_seed1_all1000_nlp_no_params_mincols.csv.gz}"
CORRECT_EXEC_CSV="${CORRECT_EXEC_CSV:-${BASE_DIR}/synth_unique_seed1_all1000_nlp_clean_child_correct_exec_mincols.csv.gz}"
CORRECT_EXEC_AND_ANSWER_CSV="${CORRECT_EXEC_AND_ANSWER_CSV:-${BASE_DIR}/synth_unique_seed1_all1000_nlp_clean_child_correct_exec_and_answer_mincols.csv.gz}"
CHUNKSIZE="${CHUNKSIZE:-200000}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "${INPUT_CSV}" ]]; then
  echo "missing input csv: ${INPUT_CSV}" >&2
  exit 1
fi

run_translate() {
  local row_filter="$1"
  local output_csv="$2"
  echo "building ${row_filter} -> ${output_csv}"
  "${PYTHON_BIN}" "${TRANSLATOR}" \
    --no-include-default-uma-traces \
    --no-include-sp2013-seeds \
    --input-csv "${INPUT_CSV}" \
    --output-csv "${output_csv}" \
    --chunksize "${CHUNKSIZE}" \
    --minimal-columns \
    --student-prompt-mode none \
    --reasoning-mode clean_child \
    --row-filter "${row_filter}"
}

run_translate "correct_exec" "${CORRECT_EXEC_CSV}"

echo "building correct_exec_and_answer -> ${CORRECT_EXEC_AND_ANSWER_CSV}"
rm -f "${CORRECT_EXEC_AND_ANSWER_CSV}"
export CORRECT_EXEC_CSV CORRECT_EXEC_AND_ANSWER_CSV CHUNKSIZE
"${PYTHON_BIN}" - <<'PY'
import os
import pandas as pd

input_csv = os.environ["CORRECT_EXEC_CSV"]
output_csv = os.environ["CORRECT_EXEC_AND_ANSWER_CSV"]
chunksize = int(os.environ.get("CHUNKSIZE", "200000"))

def truthy_mask(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip().str.lower()
    return text.isin({"1", "1.0", "true", "t", "yes", "y"})

wrote_header = False
rows_in = 0
rows_kept = 0
for chunk_idx, chunk in enumerate(pd.read_csv(input_csv, chunksize=chunksize), start=1):
    rows_in += len(chunk)
    filtered = chunk.loc[truthy_mask(chunk["is_correct"])].reset_index(drop=True)
    rows_kept += len(filtered)
    if len(filtered) == 0:
        print(f"[filter chunk {chunk_idx}] kept 0/{len(chunk):,} rows (seen {rows_in:,}, kept {rows_kept:,})", flush=True)
        continue
    mode = "w" if not wrote_header else "a"
    filtered.to_csv(output_csv, index=False, mode=mode, header=not wrote_header, compression="infer")
    wrote_header = True
    print(
        f"[filter chunk {chunk_idx}] kept {len(filtered):,}/{len(chunk):,} rows "
        f"(seen {rows_in:,}, kept {rows_kept:,})",
        flush=True,
    )

if not wrote_header:
    pd.read_csv(input_csv, nrows=0).to_csv(output_csv, index=False, mode="w", header=True, compression="infer")

print("")
print(f"Done. Filtered rows seen: {rows_in:,}")
print(f"Done. Filtered rows kept: {rows_kept:,}")
print(f"Output: {output_csv}")
PY

echo
echo "Finished building cleaned distillation datasets."
echo "  correct_exec: ${CORRECT_EXEC_CSV}"
echo "  correct_exec_and_answer: ${CORRECT_EXEC_AND_ANSWER_CSV}"

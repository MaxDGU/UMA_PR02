"""Strategy-code mappings for fraction procedural alignment.

The human SP2011 fine-tuning data uses numeric strategy codes, while UMA traces
use named strategy rules. For procedural-alignment comparisons, we normalize
both to UMA-compatible labels: `KDON`, `CDON`, `ONOD`, `ICDM`, `CROP`, and
`OTHER`.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional, Tuple

import pandas as pd


CANONICAL_STRATEGY_CODES: Tuple[str, ...] = (
    "KDON",
    "CDON",
    "ONOD",
    "ICDM",
    "CROP",
    "OTHER",
)
STRATEGY_FAMILY_CODES: Tuple[str, ...] = ("AS", "M", "D", "OTHER")

CANONICAL_STRATEGY_CODEBOOK: Mapping[str, str] = {
    "KDON": (
        "Operate on numerators, keep denominator. The answer numerator comes "
        "from applying the operation to numerators; the answer denominator is "
        "one operand denominator or the common denominator."
    ),
    "CDON": (
        "Convert to a common denominator, then operate on numerators and keep "
        "the common denominator."
    ),
    "ONOD": (
        "Operate on numerators and denominators. The answer numerator comes "
        "from applying the operation to numerators; the answer denominator "
        "comes from applying the operation to denominators."
    ),
    "ICDM": "Invert one operand, then operate on numerators and denominators.",
    "CROP": (
        "Cross-operate. Combine a numerator from one operand with a denominator "
        "from the other, and vice versa."
    ),
    "OTHER": (
        "Other, none, or guessed. Use as a last resort when the trace says the "
        "student guessed, gives no interpretable procedure, uses decimals or "
        "conceptual reasoning instead of a listed procedure, or uses a rare "
        "common-denominator plus operate-both-parts hybrid."
    ),
}

HUMAN_FRACTION_RAW_TO_CANONICAL: Mapping[str, str] = {
    "1": "KDON",
    "2": "CDON",
    "3": "ONOD",
    "4": "OTHER",
    "5": "ICDM",
    "6": "CROP",
    "10": "OTHER",
    "20": "OTHER",
}

UMA_STRATEGY_TO_CANONICAL: Mapping[str, str] = {
    "KDON_AS": "KDON",
    "KDON_OG": "KDON",
    "CDON_AS": "CDON",
    "CDON_OG": "CDON",
    "ONOD_M": "ONOD",
    "ONOD_OG": "ONOD",
    "ICDM_D": "ICDM",
    "ICDM_OG": "ICDM",
    "CROP_M": "CROP",
    "OTHER": "OTHER",
}

CANONICAL_TO_STRATEGY_FAMILY: Mapping[str, str] = {
    "KDON": "AS",
    "CDON": "AS",
    "ONOD": "M",
    "ICDM": "D",
    "CROP": "D",
    "OTHER": "OTHER",
}

CANONICAL_TO_SP2013_STRAT: Mapping[str, str] = {
    "KDON": "OpNumKeepDen",
    "CDON": "OpNumKeepDen",
    "ONOD": "IndepComp",
    "ICDM": "InvertOper",
    "CROP": "InvertOper",
    "OTHER": "Other/None",
}

SP2013_STRAT_TO_STRATEGY_FAMILY: Mapping[str, str] = {
    "OpNumKeepDen": "AS",
    "IndepComp": "M",
    "InvertOper": "D",
    "Other/None": "OTHER",
}

SP2013_CANONICAL_DESCRIPTION: Mapping[str, str] = {
    "KDON": "SP2013 `OpNumKeepDen` / AS family.",
    "CDON": "SP2013 `OpNumKeepDen` / AS family.",
    "ONOD": "SP2013 `IndepComp` / M family.",
    "ICDM": "SP2013 `InvertOper` / D family.",
    "CROP": "SP2013 `InvertOper` / D family.",
    "OTHER": "SP2013 `Other/None` / OTHER family; also merges SP2011 raw codes `4`, `10`, and `20`.",
}

_CODE_PATTERN = re.compile(r"(?<![\d.])(10|20|[1-6])(?![\d.])")
_LABEL_PATTERN = re.compile(r"\b(KDON|CDON|ONOD|ICDM|CROP|OTHER)\b", re.IGNORECASE)


def _clean_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text or None


def map_human_fraction_code_to_canonical(value: Any) -> Optional[str]:
    """Map SP2011/human-FT raw strategy codes to UMA-compatible labels.

    Human code `4` and raw `10`/`20` intentionally collapse to `OTHER`.
    Free-text values such as ``"Code 10"`` and UMA raw labels such as
    ``"KDON_AS"`` are accepted for parser robustness.
    """

    text = _clean_value(value)
    if text is None:
        return None
    upper = text.upper()
    if upper in CANONICAL_STRATEGY_CODES:
        return upper
    if upper in UMA_STRATEGY_TO_CANONICAL:
        return UMA_STRATEGY_TO_CANONICAL[upper]
    if text in HUMAN_FRACTION_RAW_TO_CANONICAL:
        return HUMAN_FRACTION_RAW_TO_CANONICAL[text]
    label_match = _LABEL_PATTERN.search(text)
    if label_match:
        return label_match.group(1).upper()
    match = _CODE_PATTERN.search(text)
    if not match:
        return None
    return HUMAN_FRACTION_RAW_TO_CANONICAL.get(match.group(1))


def map_sp2013_strat_to_canonical(value: Any) -> Optional[str]:
    """SP2013 `strat` labels do not map to unique detailed labels.

    Use `map_sp2013_strat_to_family` for SP2013 comparisons.
    """

    return None


def map_canonical_code_to_sp2013_strat(value: Any) -> Optional[str]:
    """Map a detailed canonical code to the comparable SP2013 coarse label."""

    code = map_human_fraction_code_to_canonical(value)
    if code is None:
        return None
    return CANONICAL_TO_SP2013_STRAT[code]


def map_human_fraction_code_to_family(value: Any) -> Optional[str]:
    """Map SP2011/human-FT raw strategy codes to AS/M/D/OTHER families."""

    code = map_human_fraction_code_to_canonical(value)
    if code is None:
        return None
    return CANONICAL_TO_STRATEGY_FAMILY[code]


def map_sp2013_strat_to_family(value: Any) -> Optional[str]:
    """Map SP2013 coarse `strat` labels to AS/M/D/OTHER families."""

    text = _clean_value(value)
    if text is None:
        return None
    return SP2013_STRAT_TO_STRATEGY_FAMILY.get(text)


def add_human_canonical_strategy_column(
    df: pd.DataFrame,
    source_col: str = "code",
    out_col: str = "strategy_canonical",
) -> pd.DataFrame:
    """Return a copy with canonical labels for human fine-tuning rows."""

    if source_col not in df.columns:
        raise ValueError(f"Missing human strategy code column: {source_col}")
    out = df.copy()
    out[out_col] = out[source_col].map(map_human_fraction_code_to_canonical)
    return out


def add_human_strategy_family_column(
    df: pd.DataFrame,
    source_col: str = "code",
    out_col: str = "strategy_family",
) -> pd.DataFrame:
    """Return a copy with AS/M/D/OTHER labels for human fine-tuning rows."""

    if source_col not in df.columns:
        raise ValueError(f"Missing human strategy code column: {source_col}")
    out = df.copy()
    out[out_col] = out[source_col].map(map_human_fraction_code_to_family)
    return out


def add_sp2013_canonical_strategy_column(
    df: pd.DataFrame,
    source_col: str = "strat",
    out_col: str = "strategy_canonical",
) -> pd.DataFrame:
    """Return a copy with canonical labels for SP2013 rows."""

    if source_col not in df.columns:
        raise ValueError(f"Missing SP2013 strategy label column: {source_col}")
    out = df.copy()
    out[out_col] = out[source_col].map(map_sp2013_strat_to_canonical)
    return out


def add_sp2013_strategy_family_column(
    df: pd.DataFrame,
    source_col: str = "strat",
    out_col: str = "strategy_family",
) -> pd.DataFrame:
    """Return a copy with AS/M/D/OTHER labels for SP2013 rows."""

    if source_col not in df.columns:
        raise ValueError(f"Missing SP2013 strategy label column: {source_col}")
    out = df.copy()
    out[out_col] = out[source_col].map(map_sp2013_strat_to_family)
    return out


def mapping_table_rows() -> Tuple[Dict[str, str], ...]:
    """Rows suitable for README tables or lightweight metadata exports."""

    return tuple(
        {
            "canonical_code": code,
            "strategy_family": CANONICAL_TO_STRATEGY_FAMILY[code],
            "canonical_description": CANONICAL_STRATEGY_CODEBOOK[code],
            "human_ft_raw_codes": ", ".join(
                raw
                for raw, canonical in HUMAN_FRACTION_RAW_TO_CANONICAL.items()
                if canonical == code
            ),
            "sp2013_strat": CANONICAL_TO_SP2013_STRAT[code],
            "sp2013_note": SP2013_CANONICAL_DESCRIPTION[code],
        }
        for code in CANONICAL_STRATEGY_CODES
    )

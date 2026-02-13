# Strategy Codes and OG Types (UMA Quick Guide)

This guide explains how to read UMA strategy codes in this repo and how to interpret overgeneralization (`_OG`) vs execution-level errors.

## 1) Key Terms

- `operation`: the arithmetic operator on a problem (`add`, `sub`, `mul`, `div`).
- `strategy`: the high-level method selected to solve the problem (for fractions: `KDON_*`, `CDON_*`, `ONOD_*`, `ICDM_*`, `CROP_M`).
- `execution rules`: step-level rules that carry out the selected strategy (`operate_nums`, `convert_CD`, etc.).
- `execution error rules`: step-level error variants (`convert_CD_omit_nums`, `invert_fail`, etc.).

Important: strategy-level labels and execution-level errors are different layers.

## 2) Strategy Code Format

`<PREFIX>_<SUFFIX>`

### Prefix (what procedure is used)

- `KDON`: keep denominator, operate on numerators.
- `CDON`: convert to common denominator, then KDON-like numerator operation.
- `ONOD`: operate on numerators and denominators separately.
- `ICDM`: invert and change division to multiplication.
- `CROP`: cross-operate multiplication (invert second operand, then ONOD-like multiply).

### Suffix (where/why used)

- `_AS`: add/sub version.
- `_M`: multiplication version.
- `_D`: division version.
- `_OG`: overgeneralized version (operation restriction removed from a base strategy rule).

## 3) What `_OG` Means

`_OG` is explicit in this codebase (not inferred afterward).

- `KDON_OG`, `CDON_OG`, `ONOD_OG`, `ICDM_OG` are defined as rules.
- They are built by removing operation constraints from their non-OG versions.

So OG means:
- same core procedure as the base strategy,
- but allowed to fire outside intended operations.

## 4) Examples

### In-domain (intended use)

- `KDON_AS` on `3/7 + 2/7`: `(3+2)/7 = 5/7`
- `CDON_AS` on `1/2 + 1/3`: `3/6 + 2/6 = 5/6`
- `ONOD_M` on `2/3 * 3/5`: `(2*3)/(3*5) = 6/15 = 2/5`
- `ICDM_D` on `3/4 : 2/5`: `3/4 * 5/2 = 15/8`

### Overgeneralized (out-of-domain use)

- `ONOD_OG` on addition: `2/3 + 1/3 -> (2+1)/(3+3) = 3/6` (wrong for add)
- `KDON_OG` on multiplication: `3/5 * 1/5 -> (3*1)/5 = 3/5` (wrong)
- `CDON_OG` on division: can apply common-denominator add/sub style on a division problem (wrong process for division)
- `ICDM_OG` on addition/subtraction: applying invert-and-multiply logic outside division

### `CROP_M` example

- `4/5 * 3/5` with `CROP_M`: `4/5 * 5/3 = 20/15 = 4/3` (typically wrong for multiplication)

## 5) Why OG and AS Can Both Appear on Addition

On addition/subtraction, `KDON_AS` and `KDON_OG` may execute similarly.
Same for `CDON_AS` and `CDON_OG`.

Difference:
- `*_AS` is operation-restricted.
- `*_OG` is scope-expanded (can fire where it should not).

So on add/sub, OG may still produce correct answers in some cases, but conceptually it is still an overgeneralized rule.

## 6) Overgeneralization vs Omission/Misexecution

These are not the same:

- Overgeneralization: wrong strategy scope (strategy-level).
- Omission/misexecution: steps inside a strategy are skipped/misapplied (execution-level).

Fraction execution error example:
- `convert_CD_omit_nums`: convert denominators to common denominator but omit numerator conversion.
- Example outcome: `3/5 + 1/4 -> 3/20 + 1/20 = 4/20` (wrong).

## 7) Practical Interpretation for Analysis

When analyzing errors:

- `P(strategy | error)` tells you which strategies appear most in error rows.
- `P(error | strategy)` tells you how risky each strategy is when used.
- Operation-conditioned views (`P(error | operation, strategy)`) are often best for fair comparisons.

If you include `CROP_M` in OG analyses, label it as `OG-like` explicitly, since it is overgeneralized by role but does not have `_OG` suffix.

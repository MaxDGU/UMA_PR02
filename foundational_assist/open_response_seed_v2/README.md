# FoundationalASSIST Open Response Seed V2

This folder is a compact share bundle for the cleaned FoundationalASSIST open-response seed used in the May 2026 data curation work.

## Contents

- `problems.csv`: one row per cleaned v2 problem.
- `student_answers.csv.gz`: raw row-level student `answer_text` from FoundationalASSIST interactions for the v2 problem ids, with source student ids replaced by `student_hash`.
- `answer_distribution.csv`: exact raw-answer frequency table by `problem_id` and `answer_text`.
- `manifest.json`: source paths, generation metadata, checksums, and counts.

## Provenance

The problem set comes from `symbolic-distillation/data/foundational_assist/curated/open_response_seed_v2`.

V2 was built from the deterministic `open_response_seed_v1` candidate bank by removing rows flagged in the standalone-context audit as requiring prior/group context or missing local visual/data context.

Key counts:

- Source v1 candidates: 1,335
- Excluded non-standalone candidates: 129
- V2 problems: 1,206
- Student answer rows: 690,683
- Nonempty raw answer rows: 690,681
- Clean first-attempt nonempty rows: 525,353
- Students: 5,000

## Schemas

`problems.csv` preserves the v2 problem metadata, including problem text, answer key, domain family, split, skill labels, interaction counts, and top answer surfaces.

`student_answers.csv.gz` columns:

- `problem_id`: FoundationalASSIST problem id.
- `student_hash`: SHA-256 hash of the source `user_id` with a bundle-local prefix. This supports within-bundle student grouping without exposing the source id.
- `answer_text`: raw student-entered answer text from `Interactions.csv`; intentionally not normalized. Some values contain HTML/MathML markup and embedded newlines, so use a CSV parser instead of physical line counts.
- `discrete_score`: FoundationalASSIST score for the first answer.
- `hint_count`: number of hints requested.
- `saw_answer`: whether the student requested to see the answer.
- `clean_first_attempt`: true when `hint_count == 0`, `saw_answer == false`, and `answer_text` is nonempty.

`answer_distribution.csv` columns:

- `problem_id`
- `answer_text`
- `n`
- `proportion`

## Recommended Use

Use `problems.csv` to inspect the cleaned standalone problem set. Use `answer_distribution.csv` for quick review of common student answers. Use `student_answers.csv.gz` when row-level student grouping is needed.

For student-error distribution analyses, prefer rows where `clean_first_attempt == true`.

## Privacy And License Notes

FoundationalASSIST is a gated CC-BY-NC-4.0 dataset with responsible-use requirements. This bundle is for authorized collaborators only.

Raw `answer_text` is included because it is useful for inspection, but student-entered text can theoretically contain PII. Direct source identifiers and timestamps are not included. A lightweight scan over the shared raw answers found:

- Email-like rows: 0
- URL-like rows: 518
- Rows containing `@`: 1

Most URL-like matches are MathML namespace strings embedded in raw answer markup; the scan also found a few pasted non-MathML links.

If possible PII is found during review, do not redistribute it further and follow the FoundationalASSIST responsible-use guidance.

## Limitations

This bundle is for data review and distribution analysis. The v2 canonical forms and answer checks were still marked as parser/solver pending in the source artifact, so they are not included here as validated modeling artifacts.

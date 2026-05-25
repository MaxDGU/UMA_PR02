# Qwen3-0.6B E1 Decimal BSS Traces, Best MAG6-H

Selected by six-cell `mag6_vs_human`; lower is better.

Selected temperature: `1.5`

Selected metrics:

- `mag6_vs_human`: `6.41699532747287` percentage points
- `mag6_vs_uma`: `7.74236947791165` percentage points
- `overall_acc`: `69.05%`
- `human_answer_tv_mean`: `0.250416666666667`
- rows: `12000`

Files:

- `raw_traces_temp1p5.csv`: raw selected Qwen traces.
- `trace_eval_input_temp1p5.csv`: normalized input for decimal strategy extraction from natural-language traces.
- `sixcell_metric_summary_ranked.csv`: this model's six-cell temperature grid ranked by MAG6-H.
- `manifest.json`: source paths and selection metadata.

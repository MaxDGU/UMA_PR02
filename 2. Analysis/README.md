This folder contains the data and scripts used for the analyses reported in the submission to Psychological Review.

The ".RData" file contains the output of the simulations, converted into R format.

The ".R" file contains the R script that was used to analyze the above data.

The "datasets" subfolder contains empirical data from two previous studies (Siegler & Pyke, 2013; Braithwaite et al., 2021) that were used as benchmarks for comparison to the simulation output, as described in the submission.

Python conversion helper:
- `convert_rdata_to_python.py` converts `.RData/.Rdata/.rds` files into Python-friendly outputs.
- Install dependencies once: `python -m pip install pyreadr rdata`
- Convert all R files under this repo: `python '2. Analysis/convert_rdata_to_python.py'`
- Convert one file to a custom folder: `python '2. Analysis/convert_rdata_to_python.py' '2. Analysis/sim ALL29_BCD data.Rdata' --output-dir converted_r`
- Keep arithmetic values as fraction text: `python '2. Analysis/convert_rdata_to_python.py' --as-fractions`
- Output includes per-object `.npy` files, `.npz`/`.csv` for tables, and `.meta.json` metadata.

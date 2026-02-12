# Results

Replication results for UMA (Unified Model of Arithmetic) and transformer distillation.

## Repository Structure

```
UMA_PR02/
├── 1. Model/                    # Core UMA model (uma.py, models.py, simulators.py)
│   └── Problem Sets */          # Training/testing problem sets
└── results/
    ├── UMA_replication/
    │   ├── generate_traces.py   # Script to train UMA and generate traces
    │   ├── uma_traces_all.csv.gz  # 980K reasoning traces (compressed)
    │   ├── verification_full_results.csv
    │   └── sp2013.csv           # 16-problem test set
    └── transformer_replication/
        ├── train_transformer.py # Script to train transformer on traces
        └── phase3_model.pt      # Trained model weights
```

## UMA Replication

### Files

`uma_traces_all.csv.gz` - 980K reasoning traces from 1000 simulated students. Each trace includes problem, student parameters (g, d, rt_mu, ice), strategy, goals, execution rules, and answer.

`sp2013.csv` - 16 fraction problems from Siegler & Pyke (2013) used for evaluation.

### Running from this repo

```bash
cd results/UMA_replication
python generate_traces.py
```

This trains 1000 UMA students on GoMath curriculum and generates traces. Takes ~24 hours on CPU.

Student parameter grid (1000 combinations):
- g (decision noise): 0.01 to 0.10 (10 values)
- d (error discount): 0.1 to 0.9 (5 values)
- rt_mu (retrieval threshold): 3 to 6 (4 values)
- ice (initial counting experience): 0 to 100 (5 values)

## Transformer Replication

### Files

`phase3_model.pt` - Trained transformer (decoder-only, 5 layers, 128 dim, 1.6M params).

`train_transformer.py` - Training script with curriculum learning options.

### Running from this repo

```bash
cd results/transformer_replication
python train_transformer.py --epochs 50 --batch_size 128
```

Curriculum options: `--curriculum random|blocked_complexity|blocked_ed_first|blocked_student`

### Current Results

SP2013 accuracy: 42% (vs UMA's 53%). Gap is primarily in unequal-denominator addition/subtraction where traces are 50% longer on average.

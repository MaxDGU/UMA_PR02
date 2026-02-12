# Results

This folder contains replication results for the UMA (Unified Model of Arithmetic) paper and transformer distillation experiments.

## UMA Replication

### Files

`uma_traces_all.csv.gz` - Compressed CSV containing 980K reasoning traces from 1000 simulated students solving 1000 fraction problems each. Each trace includes the problem, student parameters (g, d, rt_mu, ice), chosen strategy, goal sequence, execution rules, and final answer.

`verification_full_results.csv` - SP2013 test set results (16 fraction problems from Siegler & Pyke 2013) used to verify our UMA replication matches the original paper.

### How to generate UMA traces

1. Train UMA students on the GoMath curriculum (405 problems):

```python
from simulators import Cohort

cohort = Cohort(student_params, course)
cohort.run_all()
```

The `student_params` is a list of 1000 parameter combinations spanning:
- g (decision noise): 0.01 to 0.10
- d (error discount): 0.1 to 0.9
- rt_mu (retrieval threshold): 3 to 6
- ice (initial counting experience): 0 to 100

2. Generate reasoning traces by testing trained students on new problems:

```python
for student in cohort.students:
    for problem in test_problems:
        trace = student.solve(problem)
        # trace contains: strategy, goals, exec, answer
```

3. The full pipeline is in `replicate_uma_paper.py`. Run with:

```bash
sbatch run_replicate_paper.slurm
```

WARNING: this is a long running process but I'm finding it can be parallelized better than how I did it. Currently, this takes ~24 hours on CPU and produces the trace files.

## Transformer Replication

### Files

`phase3_model.pt` - Trained transformer weights (decoder-only, 5 layers, 128 dim, 1.6M params). Trained on UMA traces to predict reasoning sequences given student parameters and problem.

### How to train the transformer

1. Ensure you have the UMA traces file (`uma_traces_all.csv`).

2. Run training:

```bash
python train_transformer_phase3.py --epochs 50 --batch_size 128
```

Or submit as a SLURM job:

```bash
sbatch run_phase3_train.slurm
```

3. The model learns to predict: strategy, goals, execution rules, and answer given student parameters and a fraction problem.

Current accuracy on SP2013: 42% (vs UMA's 53%). The gap is primarily in unequal-denominator addition/subtraction, where the transformer struggles with longer reasoning sequences (unequal denominator requires 50% longer traces on average, but this should be ficable with better training params...).

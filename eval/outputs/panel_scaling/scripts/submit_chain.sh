#!/usr/bin/env bash
# Full chain for the 996-learner paper-recipe study. Idempotent: each stage skips finished work.
set -euo pipefail
PS=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling
N=$(wc -l < $PS/subjids_remaining.txt)
ARR=$(sbatch --parsable --array=0-$((N-1))%150 $PS/regen_one.sbatch)
T=$(sbatch --parsable --dependency=afterok:$ARR $PS/translate_all996.sbatch)
B=$(sbatch --parsable --dependency=afterok:$T $PS/build_arms.sbatch)
DB=$(bash $PS/submit_distill.sh B_all996_sub  $PS/arms/B_all996_sub520k.csv.gz 2 4 14:00:00 $B)
D42=$(bash $PS/submit_distill.sh D_rand25_s42 $PS/arms/D_rand25_s42.csv.gz    2 4 14:00:00 $B)
D43=$(bash $PS/submit_distill.sh D_rand25_s43 $PS/arms/D_rand25_s43.csv.gz    2 4 14:00:00 $B)
DC=$(bash $PS/submit_distill.sh C_all996_full $PS/arms/C_all996_full.csv.gz   8 1 2-20:00:00 $B)
H=()
for pair in "B_all996_sub:$DB" "D_rand25_s42:$D42" "D_rand25_s43:$D43" "C_all996_full:$DC"; do
  a=${pair%%:*}; d=${pair##*:}
  H+=($(sbatch --parsable --dependency=afterok:$d --job-name=hft_$a --export=ALL,ARM=$a $PS/hft_eval_arm.sbatch))
done
S=$(sbatch --parsable --dependency=afterany:$(IFS=:; echo "${H[*]}") --partition=cpu --account=griffith --time=01:00:00 --mem=16G \
   --job-name=score996 --output=$PS/logs/score_%j.out --wrap="/home/mg7411/.conda/envs/gpt-analysis/bin/python $PS/score_all.py")
echo "regen_array=$ARR translate=$T build_arms=$B distill_B=$DB distill_D42=$D42 distill_D43=$D43 distill_C=$DC hft=${H[*]} score=$S" | tee $PS/chain_jobids.txt

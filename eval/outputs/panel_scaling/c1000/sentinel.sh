#!/usr/bin/env bash
# Sentinel for arm C + arm A control: logs events every 5 min to sentinel.log and applies at most ONE automatic recovery per job.
C=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/c1000; LOG=$C/sentinel.log; S=$C/sentinel_state; mkdir -p $S
TR=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/transformer_replication; FC=/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/tmp/frac100
say(){ echo "[$(date '+%m-%d %H:%M')] $*" >> $LOG; }
get(){ grep -oP "$1=\K[0-9]+" $C/jobids.txt | tail -1; }
setid(){ echo "$1=$2" >> $C/jobids.txt; }
state(){ sacct -j $1 -X -n -o State 2>/dev/null | head -1 | awk '{print $1}'; }
say "sentinel start"; last_prog=0; last_mem=0
while true; do
  for arm in C Adrop; do
    if [[ $arm == C ]]; then D=$(get distill_C); P=$(get post_C); NAME=C1000drop; else D=$(get distill_Adrop); P=$(get post_Adrop); NAME=Adrop; fi
    st=$(state $D); prev=$(cat $S/st_$arm 2>/dev/null); [[ "$st" != "$prev" ]] && { say "$NAME distill $D: $st"; echo $st > $S/st_$arm; }
    if [[ "$st" =~ ^(OUT_OF_ME|FAILED|NODE_FAIL|TIMEOUT|PREEMPTED) ]] && [[ ! -f $S/retried_$arm ]]; then
      touch $S/retried_$arm
      if [[ $arm == C && "$st" =~ ^OUT_OF_ME ]]; then
        N=$(PARTITION=ailab ACCOUNT_ARG="--account=griffith" NEVAL=5 SAVE_EVAL=1 bash $C/submit_distill.sh C1000drop $C/corpus_parquet 4 2 2-00:00:00 ""); say "ACTION: $NAME OOM -> fresh restart on 4 GPUs as $N"
      elif [[ $arm == C ]]; then
        N=$(PARTITION=ailab ACCOUNT_ARG="--account=griffith" NEVAL=5 SAVE_EVAL=1 bash $C/submit_distill_resume.sh C1000drop $C/corpus_parquet 8 1 1-06:00:00 ""); say "ACTION: $NAME $st -> resume from last checkpoint as $N"
      else
        N=$(PARTITION=ailab ACCOUNT_ARG="--account=griffith" bash $C/submit_distill.sh Adrop $FC/regen_v10_p25/uma_frac_panel25regen_v10_nlp.csv.gz 2 4 04:00:00 ""); say "ACTION: $NAME $st -> rerun as $N"
      fi
      [[ $arm == C ]] && setid distill_C $N || setid distill_Adrop $N
      scontrol update JobId=$P Dependency=afterok:$N && say "ACTION: post array $P now depends on $N"
    elif [[ "$st" =~ ^(OUT_OF_ME|FAILED|NODE_FAIL|TIMEOUT|PREEMPTED) ]] && [[ ! -f $S/alerted_$arm ]]; then touch $S/alerted_$arm; say "ALERT: $NAME failed again ($st) after one retry; no further automatic action"; fi
    for t in $(sacct -j $P -X -n -P -o JobID,State 2>/dev/null | awk -F'|' '$2 ~ /FAILED|OUT_OF_ME|NODE_FAIL|TIMEOUT/ {split($1,a,"_"); print a[2]}'); do
      [[ -f $S/post_retried_${arm}_$t ]] && continue; touch $S/post_retried_${arm}_$t
      R=$(sbatch --parsable --array=$t --job-name=post_$NAME --export=ALL,ARM=$NAME $C/post.sbatch); say "ACTION: $NAME post task $t failed -> retry as $R"
    done
    for t in $(sacct -j $P -X -n -P -o JobID,State 2>/dev/null | awk -F'|' '$2=="COMPLETED"{split($1,a,"_"); print a[2]}'); do [[ -f $S/post_done_${arm}_$t ]] || { touch $S/post_done_${arm}_$t; say "$NAME post task $t completed"; }; done
  done
  D=$(get distill_C); f=$(ls -t $C/logs/distill_C1000drop_*.out 2>/dev/null | head -1)
  if [[ -n "$f" ]]; then
    grep -E "\[eval full" "$f" | while read l; do k=$(echo "$l" | grep -oE "e01_u[0-9]+"); [[ -f $S/eval_$k ]] || { touch $S/eval_$k; say "C EVAL $(echo "$l" | cut -c1-140)"; }; done
    grep -qiE "train_loss (nan|inf)" "$f" && [[ ! -f $S/nan ]] && { touch $S/nan; say "ALERT: NaN/inf training loss in arm C"; }
    now=$(date +%s); if (( now - last_prog > 7200 )) && [[ $(state $D) == RUNNING ]]; then last_prog=$now; u=$(grep -oE "update [0-9]+/[0-9]+ \| train_loss [0-9.]+" "$f" | tail -1); say "C progress: $u"; fi
    if (( now - last_mem > 1800 )) && [[ $(state $D) == RUNNING ]]; then last_mem=$now; m=$(sstat -j $D.batch -n -o MaxRSS 2>/dev/null | tr -d ' K'); [[ -n "$m" ]] && (( m > 470000000 )) && say "ALERT: arm C memory $((m/1048576)) GB, limit 512 GB"; fi
  fi
  for e in $C/logs/*.err; do grep -qE "Traceback|out of memory|Killed|CUDA error" $e 2>/dev/null && { b=$(basename $e); [[ -f $S/err_$b ]] || { touch $S/err_$b; say "ERROR in $b: $(grep -E 'Error|out of memory|Killed' $e | tail -1 | cut -c1-160)"; }; }; done
  for f in $C/logs/canary_[0-9]*.out; do [[ -f $f ]] || continue; b=$(basename $f); grep -q DONE $f && [[ ! -f $S/can_$b ]] && { touch $S/can_$b; say "CANARY: $(grep -E '^  ' $f | tr '\n' ' ' | cut -c1-400)"; }; done
  squeue -u mg7411 -h -o "%i %j %r" | grep -i NeverSatisfied | while read l; do say "ALERT: broken dependency: $l"; done
  ac=$(state $(get post_C)); aa=$(state $(get post_Adrop))
  if [[ -z "$(squeue -u mg7411 -h -n post_C1000drop,post_Adrop,d_C1000drop,d_Adrop 2>/dev/null)" ]] && [[ -n "$(sacct -j $(get post_C) -X -n -o State 2>/dev/null | grep COMPLETED)" ]]; then say "ALL STAGES FINISHED"; exit 0; fi
  sleep 300
done

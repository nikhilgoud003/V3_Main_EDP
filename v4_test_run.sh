#!/bin/bash
# Layer A test procedure: first files -> add new files -> rerun the new files (must change nothing) -> all files at once.
# Usage: v4_test_run.sh <test folder (new)> <first case-JSON folder> <second case-JSON folder>
#   <test folder>/run              incremental result (first + second)
#   <test folder>/run_after_pilot  state after the first folder (used by v4_consolidate_run.sh)
#   <test folder>/run_oneshot      all files in one run
#   <test folder>/timings.txt      times and the rerun check
# Set PYTHON to choose the interpreter (default python3).
set -u
if [ $# -ne 3 ]; then echo "usage: $0 <new test folder> <first json folder> <second json folder>"; exit 1; fi
H=$(cd "$(dirname "$0")" && pwd)
PY="${PYTHON:-python3}"
S="$H/v4_run_ctx.py"
FIRST=$(cd "$2" && pwd); SECOND=$(cd "$3" && pwd)
mkdir -p "$1" && cd "$1" || exit 1
for d in run run_after_pilot run_before_rerun run_oneshot all_inputs; do [ -e $d ] && { echo "$d exists"; exit 1; }; done
mkdir all_inputs && ln -s "$FIRST"/*.json all_inputs/ && ln -s "$SECOND"/*.json all_inputs/
T=timings.txt; : > $T
step() { local name=$1; shift; local s=$($PY -c 'import time;print(time.time())'); "$@"; local e=$($PY -c 'import time;print(time.time())'); echo "$name $($PY -c "print(round($e-$s,1))") s" >> $T; }
step first $PY "$S" --input "$FIRST" --output run > log_first.txt 2>&1
cp -R run run_after_pilot
step add_second $PY "$S" --input "$SECOND" --output run > log_add.txt 2>&1
cp -R run run_before_rerun
step rerun_second $PY "$S" --input "$SECOND" --output run > log_rerun.txt 2>&1
for f in entities.jsonl mentions.jsonl state.json uncertain_pairs.jsonl dropped_entities.jsonl type_cache.jsonl pending_judges.jsonl; do
  cmp -s run/$f run_before_rerun/$f && echo "rerun $f identical" >> $T || echo "rerun $f CHANGED" >> $T; done
step oneshot $PY "$S" --input all_inputs --output run_oneshot > log_oneshot.txt 2>&1
echo DONE >> $T
cat $T

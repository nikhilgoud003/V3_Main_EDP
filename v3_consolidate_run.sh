#!/bin/bash
# Layer B test procedure on a V3_test folder made by v3_test_run.sh (needs <folder>/run_after_pilot and <folder>/run).
#   1. train on the pilot state and consolidate it            -> <folder>/consolidated
#   2. snapshot                                                -> <folder>/consolidated_after_pilot
#   3. continue incrementally on pilot + 500 TXSD (same model) -> <folder>/consolidated
#   4. all at once on pilot + 500 TXSD (same model)            -> <folder>/consolidated_oneshot
#   5. real-data checks (checks.md) and the synthetic unit tests (<folder>/tests_consolidate)
# Usage: v3_consolidate_run.sh /path/to/test_folder     (the consolidated* and tests_consolidate folders must not exist)
set -u
D=$(cd "$1" && pwd)
H=$(cd "$(dirname "$0")" && pwd)
PY="${PYTHON:-python3}"   # set PYTHON to use another interpreter (e.g. a venv's python)
for f in consolidated consolidated_after_pilot consolidated_oneshot tests_consolidate; do
  if [ -e "$D/$f" ]; then echo "$D/$f already exists - use a new V3_test folder"; exit 1; fi
done
now() { "$PY" -c 'import time; print(f"{time.time():.3f}")'; }
el() { "$PY" -c "print(f'{$2 - $1:.1f}')"; }
T="$D/consolidate_timings.txt"
: > "$T"
t0=$(now); "$PY" "$H/v3_consolidate.py" --input "$D/run_after_pilot" --output "$D/consolidated" || exit 1; t1=$(now)
echo "train + consolidate pilot: $(el $t0 $t1) s" | tee -a "$T"
cp -R "$D/consolidated" "$D/consolidated_after_pilot"
t0=$(now); "$PY" "$H/v3_consolidate.py" --input "$D/run" --output "$D/consolidated" --model "$D/consolidated_after_pilot/model.joblib" || exit 1; t1=$(now)
echo "incremental, pilot + 500 TXSD: $(el $t0 $t1) s" | tee -a "$T"
t0=$(now); "$PY" "$H/v3_consolidate.py" --input "$D/run" --output "$D/consolidated_oneshot" --model "$D/consolidated_after_pilot/model.joblib" || exit 1; t1=$(now)
echo "all at once, pilot + 500 TXSD: $(el $t0 $t1) s" | tee -a "$T"
"$PY" - "$D" <<'EOF' | tee -a "$T"
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
a = [json.loads(l) for l in open(d / "consolidated_after_pilot/redirects.jsonl")]
b = [json.loads(l) for l in open(d / "consolidated/redirects.jsonl")]
print(f"earlier redirects kept: {b[:len(a)] == a} ({len(a)} after the pilot, {len(b)} after pilot + 500 TXSD)")
inc = {m["mention_id"]: m["entity_id"] for m in map(json.loads, open(d / "consolidated/mentions_resolved.jsonl"))}
one = {m["mention_id"]: m["entity_id"] for m in map(json.loads, open(d / "consolidated_oneshot/mentions_resolved.jsonl"))}
diff = [k for k in inc if inc[k] != one.get(k)]
print(f"incremental vs all at once: {len(diff)} of {len(inc)} mentions end in a different ID")
EOF
"$PY" "$H/check_consolidated.py" "$D/run" "$D/consolidated"
V3_LB_MODEL="$D/consolidated_after_pilot/model.joblib" "$PY" "$H/test_consolidate.py" "$D/tests_consolidate"

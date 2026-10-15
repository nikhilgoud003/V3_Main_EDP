#!/usr/bin/env python3
"""Real-data checks for a Layer B output (names appear here because this is a check, not code or config).

Usage: python check_consolidated.py <V3 run folder> <Layer B output folder>   -> prints and writes <output>/checks.md
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3_consolidate as V

run_dir, lb_dir = Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser()
cfg = V.load_config(V.HERE / "v3_consolidate.yaml")
nm = V.Names(cfg)
ents = {e["entity_id"]: e for e in V.read_jsonl(run_dir / "entities.jsonl")}
final = {m["layer_a_id"]: m["entity_id"] for m in V.read_jsonl(lb_dir / "mentions_resolved.jsonl")}
cons = {e["entity_id"]: e for e in V.read_jsonl(lb_dir / "entities_consolidated.jsonl")}
lines, ok_all = [], True


def out(s=""):
    print(s)
    lines.append(s)


def check(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    out(f"- **{'PASS' if ok else 'FAIL'}** {name}" + (f": {detail}" if detail else ""))


def judge_ids(pred):
    return sorted(i for i, e in ents.items() if e["entity_type"] == "judge" and any(pred(v) for v in e["name_variants"]))


out(f"# Real-data checks: {lb_dir}")
out()
out(f"Input: {run_dir}")
out()
for short, full in (("diana song", "diana song quiroga"), ("marina garcia", "marina garcia marmolejo"),
                    ("dena hanovice", "dena hanovice palermo")):
    ids = judge_ids(lambda v: v in (short, full))
    fin = sorted({final.get(i, i) for i in ids})
    check(f'"{short.title()}" + "{full.title()}" -> 1 ID', len(ids) >= 2 and len(fin) == 1,
          f"Layer A IDs {ids} -> {fin}")
ids = judge_ids(lambda v: v.split()[:1] == ["ronald"] and "morgan" in v.split())
fin = sorted({final.get(i, i) for i in ids})
check("Ronald G. Morgan -> 1 ID", len(fin) == 1, f"Layer A IDs {ids} -> {fin}; variants "
      + ", ".join(sorted({v for i in ids for v in ents[i]['name_variants']})))

fpd = sorted(i for i, e in ents.items() if "federal public defender" in (e.get("normalized_name") or "")
             or any("federal public defender" in v for v in e["name_variants"]))
by_t = collections.Counter(ents[i]["entity_type"] for i in fpd)
fpd_c = [i for i in fpd if ents[i]["entity_type"] == "counsel"]
bad = [i for i in fpd_c if cons.get(final.get(i, i), {}).get("entity_type") != "counsel"]
check("Federal Public Defender stays counsel", fpd_c and not bad,
      f"IDs named Federal Public Defender by Layer A type: {dict(by_t)}; counsel IDs still counsel after Layer B: "
      f"{len(fpd_c) - len(bad)}/{len(fpd_c)}")

# no merge joins two different full names (persons: first word 2+ letters, 2+ words; compatible = one fits the other)
viol = []
for sid, e in cons.items():
    if not e["merged_from"] or e["entity_type"] not in ("judge", "counsel", "party"):
        continue
    members = [sid] + e["merged_from"]
    if e["entity_type"] in ("party", "counsel"):   # person names only (first word a Census first name)
        firsts = [nm.core(ents[m]["normalized_name"]) for m in members]
        if not all(t and t[0] in nm.first_names for t in firsts):
            continue
    names = sorted({tuple(nm.core(v)) for m in members for v in ents[m]["name_variants"]})
    full = [t for t in names if len(t) >= 2 and len(t[0]) >= 2]
    for x in range(len(full)):
        for y in range(x + 1, len(full)):
            a, b = list(full[x]), list(full[y])
            if not (V.fits_names(a, b) or V.fits_names(b, a)):
                viol.append((e["entity_type"], sid, " ".join(a), " ".join(b)))
check("no merge joins two different full names", not viol, f"{len(viol)} incompatible name pairs inside merged IDs")
for v in viol[:40]:
    out(f"  - {v[0]} {v[1]}: \"{v[2]}\" vs \"{v[3]}\"")
out()
out(f"Result: {'ALL PASS' if ok_all else 'SOME CHECKS FAILED'}")
(lb_dir / "checks.md").write_text("\n".join(lines) + "\n")
sys.exit(0 if ok_all else 1)

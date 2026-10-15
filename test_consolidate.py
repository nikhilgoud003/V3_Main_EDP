#!/usr/bin/env python3
"""Unit tests for v3_consolidate.py (Layer B) on small synthetic V3 output folders.

Usage: python test_consolidate.py <new_test_folder>
The model under test is a saved Layer B model trained on the pilot: set V3_LB_MODEL=<model.joblib>
(default models/model.joblib next to this file). Test 7 also scans for the names in a V3 run folder when
V3_LB_CHECK_RUN=<run folder> is set. Names below are made up for the tests.
"""
import collections
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = Path(os.environ.get("V3_LB_MODEL", str(HERE / "models/model.joblib"))).expanduser()
OUT = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else None
results = []


def check(name, ok, detail=""):
    ok = bool(ok)
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)


class Synth:
    """Builds a V3 output folder (entities.jsonl, mentions.jsonl, state.json, ...) by hand."""
    PREFIX = {"judge": "SJ", "party": "SP", "counsel": "SC", "firm": "SF"}

    def __init__(self, court="zzd"):
        self.court = court
        self.ents = {}
        self.ments = []
        self.n = collections.Counter()

    def ucid(self, k, div=1):
        return f"{self.court};;{div}:16-cr-{k:05d}"

    def entity(self, etype, name, cases, side=None, is_org=False, num=None):
        self.n[etype] += 1
        eid = f"ent_{self.PREFIX[etype]}_{num if num is not None else self.n[etype]:06d}"
        clean = " ".join(re.sub(r"[^a-z0-9 ]", " ", name.lower()).split())
        e = self.ents.setdefault(eid, {"entity_id": eid, "entity_type": etype, "canonical_name": name, "normalized_name": clean,
                                       "name_variants": [clean], "courts": [self.court], "ucids": [], "n_mentions": 0,
                                       "mention_ids": []})
        if etype == "party":
            e["role_counts"] = {side or "defendant-side": 0}
        for u in cases:
            self.mention(eid, name, u, side, is_org)
        return eid

    def mention(self, eid, name, u, side=None, is_org=False):
        e = self.ents[eid]
        mid = f"mnt_{len(self.ments)}"
        clean = " ".join(re.sub(r"[^a-z0-9 ]", " ", name.lower()).split())
        m = {"raw_name": name, "clean_name": clean, "entity_type": e["entity_type"], "ucid": u, "court": self.court,
             "mention_id": mid, "context": {}, "entity_id": eid}
        if e["entity_type"] == "party":
            m["context"] = {"is_org": [is_org], "coparties": [], "batch_initials": [], "counsel_eids": [], "firm_eids": []}
            m["side"] = [side or "defendant-side"]
            e["role_counts"][side or "defendant-side"] = e["role_counts"].get(side or "defendant-side", 0) + 1
        if e["entity_type"] == "counsel":
            m["side"] = [side or "defendant-side"]
        if e["entity_type"] == "judge":
            m["context"] = {"docket": [True]}
        self.ments.append(m)
        e["mention_ids"].append(mid)
        e["n_mentions"] += 1
        if u not in e["ucids"]:
            e["ucids"].append(u)
            e["ucids"].sort()

    def write(self, d):
        d = Path(d)
        d.mkdir(parents=True)
        with open(d / "entities.jsonl", "w") as f:
            for eid in sorted(self.ents):
                f.write(json.dumps(self.ents[eid]) + "\n")
        with open(d / "mentions.jsonl", "w") as f:
            for m in self.ments:
                f.write(json.dumps(m) + "\n")
        for fn in ("uncertain_pairs.jsonl", "pending_judges.jsonl"):
            (d / fn).write_text("")
        (d / "state.json").write_text(json.dumps({"counters": dict(self.n), "processed_files": {}, "matching": "synthetic"}))
        return d


def background(s, k0, k1, div=1):
    """Ordinary cases: one defendant, one prosecutor, one defence lawyer each (made-up names)."""
    first = ["alma", "boris", "celia", "dorian", "elva", "fabian", "greta", "hector", "ines", "jasper", "kira", "lionel"]
    last = ["quenby", "ravel", "sorrel", "tavish", "umber", "vantor", "wexley", "yarrow", "zeller", "obrien", "pellam"]
    pros = s.entity("counsel", "Orla P. Vantage", [s.ucid(k, div) for k in range(k0, k1)], side="plaintiff-side")
    for k in range(k0, k1):
        u = s.ucid(k, div)
        s.entity("party", f"{first[k % len(first)].title()} {last[(k * 7) % len(last)].title()} {chr(97 + k % 26).upper()}.", [u])
        s.entity("counsel", f"{first[(k + 3) % len(first)].title()} {chr(97 + k % 26).upper()}. {last[(k * 3) % len(last)].title()}", [u])
    return pros


def build_main():
    s = Synth()
    background(s, 1, 60)
    garcia = s.entity("judge", "Guillermo R. Garcia", [s.ucid(k) for k in range(1, 13)])
    alpha = s.entity("judge", "Nancy A. Alpha", [s.ucid(k) for k in range(13, 23)])
    beta = s.entity("judge", "Nancy B. Beta", [s.ucid(k) for k in range(23, 33)])
    jsmith = s.entity("judge", "John Smith", [s.ucid(k) for k in range(33, 41)])
    jmsmith = s.entity("judge", "James Smith", [s.ucid(k) for k in range(38, 46)])
    jlane = s.entity("judge", "Robert K. Lane", [s.ucid(k) for k in range(46, 52)])
    plane = s.entity("party", "Robert K. Lane", [s.ucid(k) for k in range(46, 52)])
    guill = s.entity("judge", "Guillermo", [s.ucid(k) for k in (2, 5, 9)])
    nancy = s.entity("judge", "Nancy", [s.ucid(22), s.ucid(23)])
    return s, {"garcia": garcia, "guill": guill, "alpha": alpha, "beta": beta, "nancy": nancy, "jsmith": jsmith,
               "jmsmith": jmsmith, "jlane": jlane, "plane": plane}


def run(inp, out):
    r = subprocess.run([sys.executable, str(HERE / "v3_consolidate.py"), "--input", str(inp), "--output", str(out),
                        "--model", str(MODEL)], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:])
    return r.returncode == 0


def final_ids(out):
    return {m["layer_a_id"]: m["entity_id"] for m in map(json.loads, open(Path(out) / "mentions_resolved.jsonl"))}


def review_pairs(out):
    return {(r["id_a"], r["id_b"]) for r in map(json.loads, open(Path(out) / "review_queue.jsonl"))}


def main():
    if OUT is None or OUT.exists():
        sys.exit("usage: test_consolidate.py <new folder that does not exist yet>")
    if not MODEL.exists():
        sys.exit(f"model not found: {MODEL} (train one first: v3_consolidate.py --input <pilot run> --output <folder>)")
    OUT.mkdir(parents=True)
    s, ids = build_main()
    inp = s.write(OUT / "in_main")
    ok = run(inp, OUT / "out_main")
    check("0. Layer B runs on a synthetic V3 folder", ok)
    if not ok:
        return
    fin = final_ids(OUT / "out_main")
    rq = review_pairs(OUT / "out_main")
    g = ids
    check("1. 'Guillermo' + 'Guillermo R. Garcia' sharing cases in one division -> 1 ID",
          fin[g["guill"]] == fin[g["garcia"]], f"{fin[g['guill']]} vs {fin[g['garcia']]}")
    nancy_pairs = {tuple(sorted((g["nancy"], g["alpha"]))), tuple(sorted((g["nancy"], g["beta"])))}
    check("2. lone 'Nancy' with two judges named Nancy in the court -> not merged",
          fin[g["nancy"]] not in (fin[g["alpha"]], fin[g["beta"]]), f"Nancy -> {fin[g['nancy']]}")
    check("2b. ... and the Nancy pairs are in the review queue", nancy_pairs & rq, f"queued: {sorted(nancy_pairs & rq)}")
    check("3. 'John Smith' vs 'James Smith' judges -> never merged", fin[g["jsmith"]] != fin[g["jmsmith"]])
    check("4. a party and a judge with the same name -> never merged", fin[g["plane"]] != fin[g["jlane"]])
    ents = [json.loads(l) for l in open(OUT / "out_main/entities_consolidated.jsonl")]
    by = {e["entity_id"]: e for e in ents}
    mixed = [e["entity_id"] for e in ents if any(not x.startswith(f"ent_{e['entity_id'].split('_')[1]}_") for x in e["merged_from"])]
    check("4b. no consolidated ID mixes types", not mixed, f"{mixed[:3]}")

    # 5. determinism: same input, a second new output folder -> identical files
    ok2 = run(inp, OUT / "out_main_again")
    same = ok2 and all((OUT / "out_main" / f).read_bytes() == (OUT / "out_main_again" / f).read_bytes()
                       for f in ("entities_consolidated.jsonl", "redirects.jsonl", "mentions_resolved.jsonl",
                                 "review_queue.jsonl", "consolidation_state.json"))
    check("5. running twice on the same input -> identical output", same)
    # 5b. rerun into the same output folder -> nothing new, files unchanged
    before = {f: (OUT / "out_main" / f).read_bytes() for f in ("entities_consolidated.jsonl", "redirects.jsonl", "mentions_resolved.jsonl")}
    ok3 = run(inp, OUT / "out_main")
    check("5b. rerun on the same output folder -> no new redirects, same files",
          ok3 and all((OUT / "out_main" / f).read_bytes() == b for f, b in before.items()))

    # 6. incremental: consolidate set A, then A + B into the same folder -> earlier redirects kept
    sa, ia = build_main()
    a_dir = sa.write(OUT / "in_A")
    sb, ib = build_main()
    background(sb, 1, 30, div=2)
    newg = sb.entity("judge", "Guillermo R. Garcia", [sb.ucid(k, 2) for k in range(1, 6)])
    newf = sb.entity("judge", "G. Garcia", [sb.ucid(k, 2) for k in (2, 3)])
    sb.mention(ib["garcia"], "Guillermo R. Garcia", sb.ucid(60))   # an old ID gets a new mention
    b_dir = sb.write(OUT / "in_AB")
    okA = run(a_dir, OUT / "out_inc")
    redA = [json.loads(l) for l in open(OUT / "out_inc/redirects.jsonl")]
    finA = final_ids(OUT / "out_inc")
    okB = run(b_dir, OUT / "out_inc")
    redB = [json.loads(l) for l in open(OUT / "out_inc/redirects.jsonl")]
    finB = final_ids(OUT / "out_inc")
    st = json.loads((OUT / "out_inc/consolidation_state.json").read_text())
    check("6. consolidating A, then A + B -> earlier redirects kept (append-only)",
          okA and okB and redB[:len(redA)] == redA and all(finB[k] == v for k, v in finA.items() if v != k),
          f"{len(redA)} redirects after A, {len(redB)} after A+B; runs {len(st['runs'])}")
    r2 = st["runs"][-1]
    check("6b. the second run scores only pairs that involve new or changed IDs",
          r2["pairs_scored"] < r2["candidate_pairs"],
          f"run 2: {r2['changed_ids']} new/changed IDs, scored {r2['pairs_scored']} of {r2['candidate_pairs']} candidate pairs")
    chains = [r for r in redB if finB.get(r["surviving_id"], r["surviving_id"]) != r["surviving_id"]]
    check("6c. chained redirects collapse (every mention points at a surviving ID)",
          all(v not in {r["old_id"] for r in redB} for v in finB.values()), f"{len(chains)} chained rows")

    # 7. grep: no real person / agency / court names in code or config
    code = (HERE / "v3_consolidate.py").read_text().lower() + "\n" + (HERE / "v3_consolidate.yaml").read_text().lower()
    words = set(re.findall(r"[a-z]+", code))
    hits = []
    fjc = HERE / "data/judges_fjc.csv"
    if fjc.exists():
        import csv
        for r in csv.DictReader(open(fjc, encoding="utf-8", errors="replace")):
            full = f"{r['First Name']} {r['Last Name']}".lower().strip()
            if len(full) > 5 and full in code:
                hits.append(full)
    run_dir = Path(os.environ.get("V3_LB_CHECK_RUN", "")).expanduser()
    if os.environ.get("V3_LB_CHECK_RUN") and (run_dir / "entities.jsonl").exists():
        courts = set()
        for l in open(run_dir / "entities.jsonl"):
            e = json.loads(l)
            courts.update(e["courts"])
            n = e.get("normalized_name") or ""
            if len(n.split()) >= 2 and len(n) >= 12 and n in code:
                hits.append(n)
        hits += sorted(c for c in courts if c in words)
    check("7. no real names (FJC judges, run entities, court codes) in v3_consolidate.py / .yaml", not hits, f"{sorted(set(hits))[:10]}")
    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()

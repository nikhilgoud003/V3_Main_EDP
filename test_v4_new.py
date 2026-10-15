#!/usr/bin/env python3
"""Tests for the finish job: pending-judge scaling, name gate, organisation normalisation, office -> RELATED, export
format, readers by structure, SIGUSR1 stack dump, safe saving, and no real names in code or config.

Usage: python test_v4_new.py <new_test_folder>
Layer B checks use V4_LB_MODEL (default models/model.joblib). All names below are made up.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else None
MODEL = Path(os.environ.get("V4_LB_MODEL", str(HERE / "models/model.joblib"))).expanduser()
PY = sys.executable
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)


SCALING = r'''
import random, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[3])
import v4_run_ctx as A
n, d = int(sys.argv[1]), Path(sys.argv[2])
d.mkdir(parents=True, exist_ok=True)
A.STORAGE_DIR = d
for name, fn in (("ENTITIES_FILE", "entities.jsonl"), ("MENTIONS_FILE", "mentions.jsonl"), ("STATE_FILE", "state.json"),
                 ("UNCERTAIN_FILE", "uncertain_pairs.jsonl"), ("DROPPED_FILE", "dropped_entities.jsonl"),
                 ("PENDING_FILE", "pending_judges.jsonl"), ("GATE_REVIEW_FILE", "gate_review.jsonl"),
                 ("DOC_REVIEW_FILE", "document_review.jsonl"), ("MENTIONED_FILE", "mentioned_only.jsonl")):
    setattr(A, name, d / fn)
A.memory = A.MemoryManager(d)
rng = random.Random(7)
syl = ["ka", "lo", "mi", "ne", "ru", "ta", "vo", "zi", "be", "do", "fu", "go"]
word = lambda k: "".join(rng.choice(syl) for _ in range(k))
courts = [f"zz{c}" for c in "abcdefghij"]
names = [(word(2).title(), word(3).title()) for _ in range(max(50, n // 5))]
for i in range(n):
    first, last = rng.choice(names)
    ct = courts[hash((first, last)) % len(courts)]
    A.memory.pending.append({"raw_name": f"{first} {last}", "clean_name": f"{first} {last}".lower(), "entity_type": "judge",
                             "ucid": f"{ct};;1:16-cv-{i:06d}", "court": ct,
                             "context": {"docket": [True], "docket_title": [True], "titles": []}})
t = time.perf_counter()
stats = {}
A.resolve_pending(stats)
print(f"RESULT {n} {time.perf_counter() - t:.3f} {stats.get('judges_created_from_docket', 0)} {len(A.memory.pending)}")
'''


def scaling(t):
    script = t / "scaling.py"
    script.write_text(SCALING)
    times = {}
    for n in (1000, 10000, 50000):
        r = subprocess.run([PY, str(script), str(n), str(t / f"scale_{n}"), str(HERE)], capture_output=True, text=True)
        line = [x for x in r.stdout.splitlines() if x.startswith("RESULT")]
        if not line:
            print(r.stdout[-1500:], r.stderr[-1500:])
            return None
        _, nn, sec, created, left = line[0].split()
        times[n] = (float(sec), int(created), int(left))
    return times


def synth_cases(folder):
    folder.mkdir(parents=True)
    cases = []
    lawyer = {"email": "rq@vexxlaw.example.com", "phone": "713-555-0199", "address": "12 Oak St, Houston, TX 77002"}
    for i in range(1, 7):
        cases.append({
            "ucid": f"zzd;;4:16-cv-{i:05d}", "court": "zzd", "judge": "Magistrate Judge Alma R. Quenby",
            "referred_judges": ["Status Conference"] if i == 1 else [],
            "parties": [
                {"name": "Orla Vantage" if i % 2 else "Quenby Widgets, Inc.", "party_type": "plaintiff", "role": "Plaintiff",
                 "counsel": [{"name": "Rafe Quillon", "entity_info": {"office_name": "Vexx Law Group" if i <= 3 else "Vexx Law Group (Fresno)",
                                                                       "email": lawyer["email"] if i <= 3 else "rq2@vexxlaw.example.com",
                                                                       "phone": lawyer["phone"] if i <= 3 else "559-555-0100",
                                                                       "address": lawyer["address"] if i <= 3 else "9 Elm St, Fresno, CA 93721"}},
                             {"name": "Quenby Legal Aid Society, Inc.", "entity_info": {"email": "fdo@la.example.org", "phone": "713-555-0300"}}]},
                {"name": "Bram Ellery", "party_type": "defendant", "role": "Defendant", "counsel": []},
            ],
            "docket": [{"docket_text": "Status conference held before Magistrate Judge Alma R. Quenby."}],
        })
    for c in cases:
        (folder / (c["ucid"].replace(";;", "-").replace(":", "-") + ".json")).write_text(json.dumps(c))


def main():
    if OUT is None or OUT.exists():
        sys.exit("usage: test_v4_new.py <new folder that does not exist yet>")
    OUT.mkdir(parents=True)
    import v4_names as N
    import v4_readers as R

    # 1. scaling of the end-of-run pending-judge step
    times = scaling(OUT)
    if times:
        per = {n: times[n][0] / n for n in times}
        ratio = per[50000] / max(per[10000], 1e-9)
        check("1. pending judges: time grows ~linearly (1k / 10k / 50k pending names)",
              ratio < 2.0 and times[50000][0] < 120,
              " | ".join(f"{n}: {times[n][0]:.2f}s, {times[n][1]} judges created, {times[n][2]} left" for n in times)
              + f" | per-name time 50k vs 10k x{ratio:.2f}")
    else:
        check("1. pending judges scaling test ran", False)

    # 2. organisation normalisation: &/and/+, punctuation, line breaks, legal forms (parser), office field
    a, b, c = N.org_parts("Brown & Green, Inc."), N.org_parts("Brown and Green Inc"), N.org_parts("BROWN + GREEN, INC.")
    check("2a. '&' / 'and' / '+' / commas / periods / case -> one core name, legal form removed by the parser",
          a["core"] == b["core"] == c["core"] == "brown green", f"{a} {b} {c}")
    d1, d2 = N.org_parts("Vexx Law Group (Fresno)"), N.org_parts("Vexx Law Group - Fresno")
    d3 = N.org_parts("Vexx Law Group\nFresno Office")
    check("2b. text in parentheses or after a final dash goes to the office field",
          d1["core"] == d2["core"] == "vexx law group" and d1["office"] == d2["office"] == "fresno", f"{d1} {d2}")
    check("2c. a hyphen inside a name is not an office ('Asten-Hill Co.')", N.org_parts("Asten-Hill Co.")["office"] == "",
          f"{N.org_parts('Asten-Hill Co.')}; line break variant {d3}")
    check("2d. display names drop prefixes and keep the most complete variant",
          N.display_name(["Hon. Alma R. Quenby", "Alma Quenby"], N.PERSON) == "Alma R. Quenby"
          and N.display_name(["VEXX LAW GROUP", "Vexx Law Group\nA Professional Corporation"], N.ORG)
          == "Vexx Law Group, A Professional Corporation",
          f"{N.display_name(['Hon. Alma R. Quenby', 'Alma Quenby'], N.PERSON)!r}")

    # 3. name gate on synthetic strings (models run once per distinct string, cached)
    g = N.NameGate(OUT / "gate_cache.jsonl")
    s1 = "Status conference held before Magistrate Judge Alma R. Quenby on 2/1/2016."
    st = s1.index("Alma")
    items = {g.key("Alma R. Quenby"): ("Alma R. Quenby", None, None), g.key("Quenby Widgets, Inc."): ("Quenby Widgets, Inc.", None, None),
             g.key("Status Conference"): ("Status Conference", None, None),
             g.key("Alma R. Quenby", s1): ("Alma R. Quenby", s1, (st, st + len("Alma R. Quenby")))}
    g.vote(items)
    g.save()
    g2 = N.NameGate(OUT / "gate_cache.jsonl")
    check("3a. gate: a person name -> PERSON from both models (bare and in its sentence)",
          g.votes_of("Alma R. Quenby") == (N.PERSON, N.PERSON) and g.votes_of("Alma R. Quenby", s1)[0] == N.PERSON,
          f"{g.votes_of('Alma R. Quenby')} {g.votes_of('Alma R. Quenby', s1)}")
    check("3b. gate: a company -> ORGANISATION from both models", g.votes_of("Quenby Widgets, Inc.") == (N.ORG, N.ORG),
          f"{g.votes_of('Quenby Widgets, Inc.')}")
    check("3c. gate: a docket event is not a PERSON for either model", N.PERSON not in g.votes_of("Status Conference"),
          f"{g.votes_of('Status Conference')}")
    check("3d. gate cache: answers are reused (second gate has every string, nothing new to run)",
          g2.vote(items) == 0 and len(g2.cache) == 4, f"{len(g2.cache)} cached")

    # 4. Layer A on synthetic cases: judges must be PERSON, counsel organisations become offices, safe saving
    synth_cases(OUT / "in_a")
    proc = subprocess.Popen([PY, str(HERE / "v4_run_ctx.py"), "--input", str(OUT / "in_a"), "--output", str(OUT / "out_a")],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    lines = []
    for line in proc.stdout:                     # signal the run while it loads / runs the gate models
        lines.append(line)
        if line.startswith("Name gate:"):
            proc.send_signal(signal.SIGUSR1)
            break
    so, se = proc.communicate(timeout=600)
    check("4a. kill -USR1 prints where the run is (faulthandler) and the run still finishes",
          proc.returncode == 0 and ("File " in se or "Thread" in se or "Current thread" in se),
          f"exit {proc.returncode}, stack dump {'yes' if 'File ' in se else 'no'}")
    ments = [json.loads(l) for l in open(OUT / "out_a/mentions.jsonl")]
    dropped = [json.loads(l) for l in open(OUT / "out_a/dropped_entities.jsonl")]
    jn = {m["raw_name"] for m in ments if m["entity_type"] == "judge"}
    check("4b. judges must be PERSON: 'Status Conference' dropped with both votes logged; the real judge kept",
          "Status Conference" not in jn and any(r["raw_name"] == "Status Conference" and r.get("votes") for r in dropped)
          and any("Quenby" in x for x in jn), f"judges {sorted(jn)}")
    firm_names = {m["raw_name"] for m in ments if m["entity_type"] == "firm"}
    counsel_names = {m["raw_name"] for m in ments if m["entity_type"] == "counsel"}
    cfg_counsel = __import__("yaml").safe_load(open(HERE / "v4_rules.yaml"))["name_gate"]["counsel_organisation"]["value"]
    want_firm = cfg_counsel == "firm"
    check(f"4c. counsel that is an ORGANISATION -> {'firm / office' if want_firm else 'stays counsel (setting)'}, not dropped",
          ("Quenby Legal Aid Society, Inc." in firm_names) == want_firm and ("Quenby Legal Aid Society, Inc." in counsel_names) != want_firm,
          f"firms {sorted(firm_names)}")
    flagged = [m for m in ments if m["raw_name"] == "Quenby Legal Aid Society, Inc." and m["entity_type"] == "counsel"]
    check("4c2. with the counsel setting, that counsel mention is flagged as an organisation",
          want_firm or (flagged and all(m["context"].get("is_organisation") == [True] for m in flagged)), f"{len(flagged)} mentions")
    kinds = {m["raw_name"]: bool(m["context"].get("is_org")) for m in ments if m["entity_type"] == "party"}
    check("4d. parties: the gate decides person vs organisation",
          kinds.get("Quenby Widgets, Inc.") is True and kinds.get("Orla Vantage") is False, f"{kinds}")
    leftovers = sorted(p.name for p in (OUT / "out_a").iterdir() if p.name.endswith(".tmp"))
    check("4e. safe saving: every output written through a temp file + rename (no .tmp left)", not leftovers, f"{leftovers}")

    # 5. Layer B: office -> RELATED, export format
    if not MODEL.exists():
        check("5. Layer B model present", False, str(MODEL))
    else:
        r = subprocess.run([PY, str(HERE / "v4_consolidate.py"), "--input", str(OUT / "out_a"), "--output", str(OUT / "out_b"),
                            "--model", str(MODEL), "--export", str(OUT / "export")], capture_output=True, text=True)
        ok = r.returncode == 0
        if not ok:
            print(r.stdout[-2000:], r.stderr[-2000:])
        links = [json.loads(l) for l in open(OUT / "out_b/related_links.jsonl")] if ok else []
        ents = {json.loads(l)["entity_id"]: json.loads(l) for l in open(OUT / "out_b/entities_consolidated.jsonl")} if ok else {}
        vexx = sorted(i for i, e in ents.items() if e["entity_type"] == "firm" and "vexx" in " ".join(e["name_variants"]))
        check("5a. office: 'Vexx Law Group' and 'Vexx Law Group (Fresno)' stay two IDs, linked RELATED / office",
              ok and len(vexx) == 2 and any(l["relation"] == "office" and {l["id_a"], l["id_b"]} == set(vexx) for l in links),
              f"firm IDs {vexx}, links {[(l['id_a'], l['id_b'], l['relation']) for l in links]}")
        files = sorted(p.name for p in (OUT / "export").iterdir()) if ok else []
        ee = [json.loads(l) for l in open(OUT / "export/entities.jsonl")] if ok else []
        em = [json.loads(l) for l in open(OUT / "export/mentions.jsonl")] if ok else []
        final_ids = {e["entity_id"] for e in ee}
        party = [e for e in ee if e["entity_type"] == "party"]
        check("5b. export writes exactly entities.jsonl and mentions.jsonl",
              files == ["entities.jsonl", "mentions.jsonl"], f"{files}")
        check("5c. export entities: variants, courts, cases; parties carry plaintiff / defendant counts",
              ee and all({"entity_id", "entity_type", "canonical_name", "name_variants", "courts", "cases"} <= set(e) for e in ee)
              and party and all("plaintiff_side" in e and "defendant_side" in e for e in party),
              f"{len(ee)} entities, e.g. {party[0] if party else None}")
        oc = [e for e in ee if e["entity_type"] == "counsel" and "quenby legal aid society" in " ".join(e["name_variants"])]
        om = [m for m in em if m["raw_name"] == "Quenby Legal Aid Society, Inc."]
        check("5e. export carries the organisation flag for counsel (entities and mentions)",
              oc and all(e.get("is_organisation") is True for e in oc) and om and all(m.get("is_organisation") is True for m in om),
              f"{[(e['entity_id'], e.get('is_organisation')) for e in oc]}")
        check("5d. export mentions: every mention, with role / side and a FINAL entity ID",
              em and all({"mention_id", "raw_name", "case", "role", "side", "entity_id"} <= set(m) for m in em)
              and all(m["entity_id"] in final_ids for m in em if m["entity_id"]),
              f"{len(em)} mentions")

    # 6. readers chosen by structure, not name
    check("6. readers by structure: docket / OCR document / generic JSON",
          R.file_kind({"ucid": "zzd;;1:16-cv-00001", "parties": []}) == "docket"
          and R.file_kind({"filename": "x", "pages": [{"page": 1, "blocks": [{"text": "a"}]}]}) == "document"
          and R.file_kind({"anything": ["else"]}) == "generic"
          and R.case_from_filename("zzd;;1-16-cv-00001_4_output.json") == ("zzd", "1:16-cv-00001")
          and R.case_from_text(["Case: 1:16-cv-00001 Document #: 4"]) == "1:16-cv-00001", "")

    # 7. no real names in code or config (FJC judge names, court codes)
    import csv
    import re
    code = "\n".join((HERE / f).read_text().lower() for f in ("v4_run_ctx.py", "v4_rules.yaml", "v4_names.py", "v4_readers.py",
                                                            "v4_consolidate.py", "v4_consolidate.yaml"))
    hits = []
    fjc = HERE / "data/judges_fjc.csv"
    courts = set()
    if fjc.exists():
        for r in csv.DictReader(open(fjc, encoding="utf-8", errors="replace")):
            full = f"{r['First Name']} {r['Last Name']}".lower().strip()
            if len(full) > 5 and re.search(r"\b" + re.escape(full) + r"\b", code):
                hits.append(full)
    for p in sorted(Path(os.environ.get("V4_PILOT_DIR", "/nonexistent")).glob("*.json"))[:2000]:
        courts.add(p.name.split("-")[0])
    words = set(re.findall(r"[a-z]+", code))
    hits += sorted(c for c in courts if c in words)
    check("7. no real names in v4_run_ctx.py / v4_rules.yaml / v4_names.py / v4_readers.py / v4_consolidate.*",
          not hits, f"{sorted(set(hits))[:10]}" + ("" if courts else " (court codes not checked: set V4_PILOT_DIR)"))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()

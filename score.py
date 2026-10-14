#!/usr/bin/env python3
"""Score runs against the pilot answer key folder (--key or $V4_ANSWER_KEY; the key is not in the repo).

Usage:
  python3 score.py --v2 DIR --v3 DIR [--key DIR] [--out score.json]

Both systems are first read into one format: rows of
  (entity_type, file, name, system_id)
  V2: judge_disambiguation.jsonl (SJID), party_disambiguation.jsonl (SPID_Strong, listed
      parties only, entity_source == "party"), firm_disambiguation.jsonl (sfid_strong; a row
      with id -1 has no ID and is its own singleton).
  V3: mentions.jsonl + entities.jsonl (entity id); party_alias mentions left out, like V2's
      alias rows, because the key lists only the parties themselves.
Only rows in the 200 key files count.

Name matching (identical for V2 and V3): a system name matches a key entity of the same type
in the same file when, after normalization (accents, case, punctuation, "&"->"and", leading
titles such as Judge/Magistrate/Honorable, trailing legal forms such as Inc/LLC/LLP/PC,
"U.S."/"United States" -> "us"), it is equal to one of the key entity's names, or
rapidfuzz token_set_ratio >= 90 with one of them and the shorter name has at least half the
words of the longer one; for judges the key judge's surname must also be in the system name.
The best-scoring key entity wins.

a) Extraction (per type)
   precision = matched system names / system names      (distinct (file, normalized name))
   recall    = key entities found / key entities         (key rows = (file, gold_id))
b) Resolution (per type, pairwise) over the found items, one item per distinct
   (file, system_id, gold_id):
   precision = pairs with same system_id AND same gold_id / pairs with same system_id
   recall    = pairs with same system_id AND same gold_id / pairs with same gold_id
95% ranges: 1,000 bootstrap resamples of the 200 files (seed 42), 2.5th-97.5th percentile.
"""

import argparse
import collections
import csv
import json
import os
import random
import re
import unicodedata
from pathlib import Path

from rapidfuzz import fuzz

TITLES = {"judge", "judges", "magistrate", "honorable", "hon", "chief", "senior", "sr", "district", "the", "mag",
          "visiting", "bankruptcy", "circuit", "us", "united", "states", "justice"}
LEGAL = {"inc", "incorporated", "llc", "llp", "lp", "corp", "corporation", "co", "company", "ltd", "limited", "plc",
         "pllc", "pc", "psc", "pa", "na", "ag", "sa", "chartered", "chtd", "lpa", "et", "al"}
GOV = {"usa", "us", "us of america", "united states", "united states of america", "u s a"}


def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s if not unicodedata.combining(c))


def norm(name, etype):
    s = fold(name).lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = s.replace("&", " and ").replace("+", " and ").replace("'", "").replace("’", "")
    s = re.sub(r"\bu\.?\s?s\.?a?\b\.?", "us", s)
    s = s.replace("united states", "us")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    t = s.split()
    if etype == "judge":
        while t and t[0] in TITLES:
            t.pop(0)
    else:
        if t and t[0] == "the":
            t.pop(0)
        t = [x for x in t if x != "and"]
        while t and t[-1] in LEGAL:
            t.pop()
    out = " ".join(t)
    if etype == "party" and out in {"us", "us of america", "usa"}:
        out = "us"
    return out


# ---------------------------------------------------------------- loaders
def ucid_to_file(ucid):
    court, cid = ucid.split(";;")
    a, rest = cid.split(":", 1)
    return f"{court}-{a}-{rest.replace(':', '-')}.json"


def load_v2(d):
    d = Path(d)
    rows = []
    for t, fn, idk, namek in (("judge", "judge_disambiguation.jsonl", "SJID", "Extracted_Entity"),
                              ("party", "party_disambiguation.jsonl", "SPID_Strong", "party_name"),
                              ("firm", "firm_disambiguation.jsonl", "sfid_strong", "raw_office_name")):
        for i, line in enumerate(open(d / fn)):
            r = json.loads(line)
            if t == "party" and r.get("entity_source") != "party":
                continue
            sid = r.get(idk)
            if sid in (None, "", -1, "-1"):
                sid = f"noid:{t}:{i}"
            rows.append((t, ucid_to_file(r["ucid"]), r.get(namek) or "", f"V2:{sid}"))
    return rows


def load_v3(d):
    d = Path(d)
    m2e = {}
    for line in open(d / "entities.jsonl"):
        e = json.loads(line)
        for mid in e.get("mention_ids") or []:
            m2e[mid] = e["entity_id"]
    rows = []
    for line in open(d / "mentions.jsonl"):
        m = json.loads(line)
        t = m.get("entity_type") or m.get("type")
        if t not in ("judge", "party", "firm") or m.get("docket_source") == "party_alias":
            continue
        sid = m2e.get(m["mention_id"]) or f"noid:{m['mention_id']}"
        rows.append((t, m.get("source_file") or ucid_to_file(m["ucid"]), m.get("raw_name") or "", f"V3:{sid}"))
    return rows


def _jsonl_rows(d):
    for fp in sorted(Path(d).rglob("*.jsonl")):
        for line in open(fp):
            if line.strip():
                yield json.loads(line)


def load_v1(d):
    """V1 run folder: judges/build/SEL_Annotations (one row per mention, SJID; after JED_3) and
    firms/out/Firm_Disambiguations (office_name, SFID_Strong). V1 parties were not run."""
    d = Path(d)
    rows = []
    for t, sub, idk, namek in (("judge", "judges/build/SEL_Annotations", "SJID", "Extracted_Entity"),
                               ("firm", "firms/out/Firm_Disambiguations", "SFID_Strong", "office_name")):
        for i, r in enumerate(_jsonl_rows(d / sub)):
            sid = r.get(idk)
            if sid in (None, "", -1, "-1"):
                sid = f"noid:{t}:{i}"
            rows.append((t, ucid_to_file(r["ucid"]), r.get(namek) or "", f"V1:{sid}"))
    return rows


def load_v4(fp):
    """V4 entities file (no mentions file): one row per (entity, ucid, name variant).
    Exact here because no entity has both >1 name variant and >1 ucid. Counsel not scored."""
    rows = []
    for line in open(fp):
        if not line.strip():
            continue
        e = json.loads(line)
        t = e.get("entity_type")
        if t not in ("judge", "party", "firm"):
            continue
        sid = e.get("entity_id") or f"noid:{t}:{e.get('canonical_name')}"
        names = e.get("name_variants") or [e.get("normalized_name") or e.get("canonical_name") or ""]
        for u in e.get("ucids") or []:
            for n in names:
                rows.append((t, ucid_to_file(u), n, f"V4:{sid}"))
    return rows


def load_v4dir(d):
    """V4 run folder (v4_run.py / v4_run_ctx.py): mentions.jsonl rows carry entity_id. Counsel not scored."""
    rows = []
    for line in open(Path(d) / "mentions.jsonl"):
        if not line.strip():
            continue
        m = json.loads(line)
        t = m.get("entity_type")
        if t not in ("judge", "party", "firm"):
            continue
        # mentions in an "unresolved" judge record are not one identity: score each as its own singleton
        sid = f"noid:{m['mention_id']}" if m.get("unresolved") else m["entity_id"]
        rows.append((t, ucid_to_file(m["ucid"]), m.get("raw_name") or "", f"V4dir:{sid}"))
    return rows


def load_key(d):
    d = Path(d)
    key = collections.defaultdict(lambda: collections.defaultdict(dict))  # type -> file -> gold -> {names}
    for r in csv.DictReader(open(d / "answer_key_detail.csv")):
        ev = r["evidence"]
        forms = []
        m = re.search(r"(?:forms|names in file): (.*?)(?:; (?:addr|domain)|$)", ev)
        if m:
            forms = [x.strip() for x in m.group(1).split(" | ") if x.strip()]
        e = key[r["entity_type"]][r["file"]].setdefault(r["gold_id"], {"names": set(), "canonical": r["name"]})
        e["names"].add(r["name"])
        e["names"].update(forms)
    return key


# ---------------------------------------------------------------- matching
def surname(name):
    t = [x for x in norm(name, "judge").split() if x not in {"jr", "sr", "ii", "iii", "iv"}]
    return t[-1] if t else ""


def build_matcher(key):
    cache = {}
    prepared = {t: {f: {g: ([norm(n, t) for n in e["names"] if norm(n, t)], surname(e["canonical"]))
                        for g, e in gs.items()} for f, gs in fs.items()} for t, fs in key.items()}

    def match(t, f, name):
        k = (t, f, name)
        if k in cache:
            return cache[k]
        s = norm(name, t)
        best, bs = None, 0
        if s:
            for g, (forms, sur) in prepared.get(t, {}).get(f, {}).items():
                if t == "judge" and sur and sur not in s.split():
                    continue
                for fm in forms:
                    if s == fm:
                        sc = 101
                    else:
                        a, b = s.split(), fm.split()
                        if min(len(a), len(b)) / max(len(a), len(b)) < 0.5:
                            continue
                        sc = fuzz.token_set_ratio(s, fm)
                        if sc < 90:
                            continue
                    if sc > bs:
                        best, bs = g, sc
        cache[k] = best
        return best

    return match


# ---------------------------------------------------------------- scoring
def prf(tp_p, n_p, tp_r, n_r):
    p = tp_p / n_p if n_p else 0.0
    r = tp_r / n_r if n_r else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def c2(n):
    return n * (n - 1) / 2


def prepare(t, rows, key, match, files):
    """Match once: distinct system names (file, ok), key rows (file, found), resolution items."""
    sys_items, found, items = {}, set(), set()
    for (rt, f, name, sid) in rows:
        if rt != t or f not in files:
            continue
        g = match(t, f, name)
        k = (f, norm(name, t))
        sys_items[k] = sys_items.get(k, False) or bool(g)
        if g:
            found.add((f, g))
            items.add((f, sid, g))
    key_rows = [(f, g, (f, g) in found) for f, gs in key[t].items() if f in files for g in gs]
    return list(sys_items.items()), key_rows, sorted(items)


def aggregate(prep, w):
    sys_items, key_rows, items = prep
    sys_n = sum(w.get(f, 0) for (f, _), ok in sys_items)
    sys_ok = sum(w.get(f, 0) for (f, _), ok in sys_items if ok)
    key_n = sum(w.get(f, 0) for f, g, ok in key_rows)
    key_found = sum(w.get(f, 0) for f, g, ok in key_rows if ok)
    ext = prf(sys_ok, sys_n, key_found, key_n)
    # Pairs of distinct items; an item from a file drawn k times has weight k, and a pair
    # (i, j) counts w_i * w_j. Per group: (S^2 - sum of w^2) / 2. With all weights 1 this is C(n, 2).
    def pairs(groups):
        tot = 0.0
        for ws in groups.values():
            s_ = sum(ws)
            tot += (s_ * s_ - sum(x * x for x in ws)) / 2
        return tot

    by_sid, by_gold, by_both = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    for (f, sid, g) in items:
        k = w.get(f, 0)
        if k:
            by_sid[sid].append(k)
            by_gold[g].append(k)
            by_both[(sid, g)].append(k)
    tp = pairs(by_both)
    ps, pg = pairs(by_sid), pairs(by_gold)
    res = prf(tp, ps, tp, pg)
    counts = {"system_names": sys_n, "system_names_matched": sys_ok, "key_entities": key_n,
              "key_entities_found": key_found, "resolution_items": sum(w.get(f, 0) for f, _, _ in items),
              "pairs_true_positive": tp, "pairs_same_system_id": ps, "pairs_same_gold_id": pg}
    return ext, res, counts


def bootstrap(prep, files, reps=1000, seed=42):
    rng = random.Random(seed)
    files = sorted(files)
    ext_s, res_s = [], []
    for _ in range(reps):
        w = collections.Counter(rng.choice(files) for _ in files)
        e, r, _ = aggregate(prep, w)
        ext_s.append(e)
        res_s.append(r)

    def ci(vals, i):
        v = sorted(x[i] for x in vals)
        return [round(v[int(0.025 * len(v))], 3), round(v[int(0.975 * len(v)) - 1], 3)]

    return ({k: ci(ext_s, i) for i, k in enumerate("prf")}, {k: ci(res_s, i) for i, k in enumerate("prf")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", help="V2 run folder (optional)")
    ap.add_argument("--v3", help="V3 run folder (optional)")
    ap.add_argument("--v3q", help="second V3 run (e.g. --qwen), scored as label V3q")
    ap.add_argument("--v1", help="V1 run folder (judges + firms only)")
    ap.add_argument("--v4", help="V4 entities .jsonl file")
    ap.add_argument("--v4dir", action="append", default=[], help="LABEL=DIR of a V4 run folder with mentions.jsonl (repeatable)")
    ap.add_argument("--key", default=os.environ.get("V4_ANSWER_KEY"), required=not os.environ.get("V4_ANSWER_KEY"),
                    help="answer key folder (default: $V4_ANSWER_KEY)")
    ap.add_argument("--out", default="score.json")
    ap.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args()
    key = load_key(a.key)
    files = set(open(Path(a.key) / "work/sample200.txt").read().split())
    match = build_matcher(key)
    out = {}
    systems = []
    if a.v2:
        systems.append(("V2", load_v2(a.v2), ("judge", "party", "firm")))
    if a.v3:
        systems.append(("V3", load_v3(a.v3), ("judge", "party", "firm")))
    if a.v3q:
        systems.append(("V3q", load_v3(a.v3q), ("judge", "party", "firm")))
    if a.v1:
        systems.insert(0, ("V1", load_v1(a.v1), ("judge", "firm")))
    if a.v4:
        systems.append(("V4", load_v4(a.v4), ("judge", "party", "firm")))
    for spec in a.v4dir:
        label, _, d = spec.partition("=")
        systems.append((label, load_v4dir(d), ("judge", "party", "firm")))
    for label, rows, types in systems:
        out[label] = {}
        for t in types:
            prep = prepare(t, rows, key, match, files)
            ext, res, counts = aggregate(prep, {f: 1 for f in files})
            ci_e, ci_r = bootstrap(prep, files, a.reps)
            out[label][t] = {
                "extraction": {"precision": round(ext[0], 3), "recall": round(ext[1], 3), "f1": round(ext[2], 3), "ci95": ci_e},
                "resolution": {"precision": round(res[0], 3), "recall": round(res[1], 3), "f1": round(res[2], 3), "ci95": ci_r},
                "counts": counts,
            }
    Path(a.out).write_text(json.dumps(out, indent=1))
    for label in out:
        for t in out[label]:
            e, r = out[label][t]["extraction"], out[label][t]["resolution"]
            print(f"{label} {t:5s} extraction P {e['precision']:.3f} R {e['recall']:.3f} F1 {e['f1']:.3f} {e['ci95']['f']} | "
                  f"resolution P {r['precision']:.3f} R {r['recall']:.3f} F1 {r['f1']:.3f} {r['ci95']['f']}")


if __name__ == "__main__":
    main()

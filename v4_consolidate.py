#!/usr/bin/env python3
"""Layer B: learned consolidation of a V4 (v4_run_ctx.py) output folder.

Runs after v4_run_ctx.py and merges IDs that belong to the same real person or organisation. The decision comes from
a learned, calibrated pair scorer, not from hand-written matching rules:

  1. one entity card per ID (names, type, courts, divisions, cases, case years, titles, party sides, co-occurring
     entities, mention count)
  2. candidate pairs (speed filter only): same type AND (share a case OR same court with char n-gram TF-IDF cosine
     above a data-derived low cut)
  3. generic pair features (FEATURES below)
  4. LogisticRegression and HistGradientBoosting, isotonic-calibrated, court group k-fold; the better one on held-out
     courts is saved with its thresholds (model.joblib) and can be reused with --model
  5. labels: the pilot answer key (score.py), the FJC list, and automatic fragments of confident full names
  6. merge if p >= T_high, review queue if T_low <= p < T_high, else keep apart
  7. merges in descending score; refused if they mix types, join two FJC judges, or join two different full first
     names with the same surname; surviving ID = the oldest ID
  8. incremental: redirects are append-only; later runs score only pairs that involve new or changed IDs
  9. --llm local|api (off by default): reviews queue pairs only after its precision is measured

Usage:
  python v4_consolidate.py --input <v4 output> --output <new folder> [--model <model.joblib>] [--llm local|api]
                           [--cases <case json folder> ...] [--config v4_consolidate.yaml] [--key <answer key dir>]
Without --model, a model is trained on --input (labels from the answer key, the FJC list and fragments) and saved
as <output>/model.joblib. Running again on the same --output continues incrementally from consolidation_state.json.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
import unicodedata
import urllib.request
import zlib
from pathlib import Path

import joblib
import numpy as np
import yaml
from rapidfuzz.distance import JaroWinkler
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, precision_recall_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v4_names as N  # noqa: E402  (shared organisation normalisation and display names)

TYPES = ("judge", "party", "counsel", "firm")
MODEL_VERSION = "layer-b-2"
ORG_KINDS = ("firm", "party_org")

FEATURES = [
    "cos",             # char n-gram TF-IDF cosine of the main names
    "jw",              # best Jaro-Winkler similarity over name variants
    "prefix",          # one name = the first words of the other (fullest name of each ID)
    "subset",          # the shorter name's words appear in order in the longer one (an initial of the shorter may stand for a word)
    "first_compat",    # first words: 1 equal, 0.75 initial vs word with that letter, 0 different
    "surname_match",   # last words equal (0.5 when a name has one word)
    "word_to_initial", # words of the shorter name that only match an initial of the longer one
    "suffix_conflict", # both names carry a generation suffix (jr, sr, iii, ...) and they differ
    "one_word",        # either name is one word
    "both_one_word",
    "ntok_short",      # words in the shorter / longer name
    "ntok_long",
    "case_jaccard",    # shared-case Jaccard
    "shared_cases",    # log(1 + shared cases)
    "same_court",
    "same_division",   # share a division (number before ":" in the case ID)
    "co_jaccard",      # Jaccard of co-occurring entities (any type, same cases)
    "co_shared",       # log(1 + shared co-occurring entities)
    "date_overlap",    # case-year ranges overlap
    "date_gap",        # years between the ranges (0 if they overlap)
    "title_compat",    # judge titles: 1 shared, 0 disjoint, 0.5 unknown
    "side_agree",      # cosine of party-side counts (0.5 unknown)
    "count_ratio",     # mention-count ratio (small / large)
    "count_min_log",   # log(1 + mentions of the smaller ID)
    "comp_court",      # other IDs of this type in the court that also fit the shorter name
    "comp_div",        # ... of those, in the same division as the shorter name's ID
    "is_judge", "is_party", "is_counsel", "is_firm",
    "is_person",
    # organisations (0 for person pairs); names from v4_names.org_parts, the same normalisation as Layer A
    "org_pair",           # both IDs are organisations
    "org_same_core",      # same normalised core name (legal forms removed)
    "org_same_loose",     # same core once the parser's organisation-type words are removed (Company, Corp., ...)
    "org_office_differs", # same core, different office (text in parentheses / after a final dash)
    "org_core_prefix",    # one core is the first words of the other
    "org_shared_counsel", # log(1 + counsel IDs linked to both: the same lawyers used both names)
    "org_shared_domain",  # a shared email domain
    "org_shared_phone",
    "org_shared_address",
]


# ---------------------------------------------------------------- config / io
def load_config(path):
    cfg = yaml.safe_load(open(path))
    cfg["_dir"] = Path(path).resolve().parent
    return cfg


def cfg_path(cfg, p):
    p = Path(os.path.expanduser(p))
    return p if p.is_absolute() else cfg["_dir"] / p


def read_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def write_jsonl(p, rows):
    """Temp file + rename, so an interrupted run never leaves a half-written output."""
    p = Path(p)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    os.replace(tmp, p)


def id_num(eid):
    m = re.search(r"(\d+)$", eid)
    return int(m.group(1)) if m else 0


def age_key(eid):
    """Oldest first: the counter number inside a type, then the ID text."""
    return (id_num(eid), eid)


def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s if not unicodedata.combining(c))


def norm_text(s):
    s = fold(s).lower().replace("'", "").replace("’", "")
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s).split())


# ---------------------------------------------------------------- names
class Names:
    def __init__(self, cfg):
        self.suffixes = set(cfg["names"]["suffixes"])
        self.first_names = set()
        for k in ("male_first", "female_first"):
            fp = cfg_path(cfg, cfg["names"][k])
            if fp.exists():
                for line in open(fp, encoding="utf-8", errors="replace"):
                    if line.split():
                        self.first_names.add(line.split()[0].lower())

    def suffix(self, name):
        return {t for t in name.split()[1:] if t in self.suffixes}

    def core(self, name):
        t = name.split()
        while len(t) > 1 and t[-1] in self.suffixes:
            t.pop()
        return t


def compat(x, y):
    return x == y or (len(x) == 1 and y.startswith(x)) or (len(y) == 1 and x.startswith(y))


def ordered_fit(ts, tc, strict=False):
    """ts (shorter) appears in order inside tc, word by word, initials allowed. strict: only an initial of ts may stand
    for a word of tc (a full word of ts is never explained by an initial of tc)."""
    j = 0
    for x in ts:
        while j < len(tc) and not (compat(x, tc[j]) and (not strict or len(tc[j]) > 1 or len(x) == 1)):
            j += 1
        if j == len(tc):
            return False
        j += 1
    return True


def fit_initials(ts, tc):
    """Ordered fit of ts in tc; returns how many full words of ts matched only an initial in tc (None if no fit)."""
    j, n = 0, 0
    for x in ts:
        while j < len(tc) and not compat(x, tc[j]):
            j += 1
        if j == len(tc):
            return None
        n += 1 if (len(x) > 1 and len(tc[j]) == 1) else 0
        j += 1
    return n


def fits_names(ts, tc):
    if not ts or not tc or len(ts) > len(tc):
        return False
    if len(ts) == 1:
        return ts[0] in (tc[0], tc[-1])
    return compat(ts[0], tc[0]) and ordered_fit(ts, tc)


def full_firsts(card):
    """surname -> set of full first names (2+ letters) over the card's name variants (persons only)."""
    out = collections.defaultdict(set)
    if not card["is_person"]:
        return out
    for t in card["toks"]:
        if len(t) >= 2 and len(t[0]) >= 2:
            out[t[-1]].add(t[0])
    return out


# ---------------------------------------------------------------- cards
def case_parts(ucid, pivot):
    court, _, cid = ucid.partition(";;")
    m = re.match(r"(\d+):(\d{2})-", cid)
    if not m:
        return court, f"{court}:?", None
    yy = int(m.group(2))
    return court, f"{court}:{m.group(1)}", (2000 + yy if yy < pivot else 1900 + yy)


def layer_a_placeholder_re(cfg):
    """Layer A's own placeholder rule (v4_rules.yaml doe_pattern + placeholder_party), read only. Layer A never merges
    such party names across cases; Layer B leaves those IDs alone so it never undoes that decision."""
    p = cfg_path(cfg, cfg["reference"]["layer_a_rules"])
    try:
        r = yaml.safe_load(open(p))
    except Exception:
        return None
    parts = []
    if r.get("doe_pattern", {}).get("pattern"):
        parts.append(r["doe_pattern"]["pattern"].strip())
    ph = r.get("placeholder_party") or {}
    if ph.get("words"):
        parts.append(r"\b(?:" + "|".join(re.escape(w) for w in ph["words"]) + r")\b")
    if ph.get("exact_names"):
        parts.append("^(?:" + "|".join(re.escape(w) for w in ph["exact_names"]) + ")$")
    return re.compile("|".join(f"(?:{x})" for x in parts)) if parts else None


class World:
    """Entity cards of one V4 output folder (plus optional synthetic fragment cards for training)."""

    def __init__(self, cfg, entities, mentions):
        self.cfg = cfg
        self.nm = Names(cfg)
        ph = layer_a_placeholder_re(cfg)
        self.placeholders = set()
        pivot = cfg["case_numbers"]["year_pivot"]
        self.cards = {}
        for e in entities:
            if e.get("unresolved") or e.get("entity_type") not in TYPES:
                continue
            names = set(e.get("name_variants") or []) | {e.get("normalized_name") or ""}
            names = sorted(n for n in names if n.strip())
            if not names:
                continue
            main = e.get("normalized_name") or names[0]
            if ph is not None and e["entity_type"] == "party" and any(ph.search(n) for n in names):
                self.placeholders.add(e["entity_id"])
                continue
            self.cards[e["entity_id"]] = {
                "id": e["entity_id"], "type": e["entity_type"], "canonical": e.get("canonical_name") or main,
                "main": main, "names": names, "cases": set(e.get("ucids") or []), "n": e.get("n_mentions") or 0,
                "titles": set(), "sides": collections.Counter(), "roles": collections.Counter(e.get("role_counts") or {}),
                "org": [0, 0], "raw": [], "synthetic": False, "source": None}
        rel_counsel = collections.defaultdict(set)
        contact = collections.defaultdict(lambda: {"domains": set(), "phones": set(), "addresses": set()})
        raws = collections.defaultdict(collections.Counter)
        for m in mentions:
            ctx = m.get("context") or {}
            if m.get("entity_type") == "counsel" and m.get("entity_id"):
                doms = {e.split("@", 1)[1] for e in ctx.get("emails") or [] if "@" in e}
                for f in ctx.get("firm_eids") or []:
                    rel_counsel[f].add(m["entity_id"])
                for pid in ctx.get("party_eids") or []:
                    rel_counsel[pid].add(m["entity_id"])
                    contact[pid]["domains"] |= doms
                    contact[pid]["phones"] |= {x for x in ctx.get("phones") or [] if x}
                    contact[pid]["addresses"] |= {x for x in ctx.get("addresses") or [] if x}
            elif m.get("entity_type") == "firm" and m.get("entity_id"):
                for k in ("domains", "phones", "addresses"):
                    contact[m["entity_id"]][k] |= {x for x in ctx.get(k) or [] if x}
            c = self.cards.get(m.get("entity_id"))
            if c is None or m.get("unresolved"):
                continue
            if m.get("entity_type") == "party":
                rel_counsel[c["id"]] |= set(ctx.get("counsel_eids") or [])
            c["titles"].update(ctx.get("titles") or [])
            c["sides"].update(m.get("side") or [])
            org_vote = bool(ctx.get("is_org")) or str((ctx.get("gate") or [""])[0]).startswith(N.ORG)
            c["org"][1 if org_vote else 0] += 1
            if m.get("raw_name"):
                raws[c["id"]][m["raw_name"]] += 1
                if len(c["raw"]) < 5 and m["raw_name"] not in c["raw"]:
                    c["raw"].append(m["raw_name"])
        for i, c in self.cards.items():
            c["rel_counsel"] = rel_counsel.get(i, set())
            ct = contact.get(i) or {"domains": set(), "phones": set(), "addresses": set()}
            c["rel_domains"], c["rel_phones"], c["rel_addresses"] = ct["domains"], ct["phones"], ct["addresses"]
            parts = collections.Counter()
            for r, n in raws.get(i, {}).items():
                op = N.org_parts(r)
                parts[(op["core"], op["office"], op["loose"])] += n
            best = max(parts.items(), key=lambda kv: (kv[1], kv[0]))[0] if parts else (c["main"], "", c["main"])
            c["org_core"], c["org_office"], c["org_loose"] = best
            c["raw_all"] = sorted(raws.get(i, {}))
        for c in self.cards.values():
            self._finish(c, pivot)
        self.case_index = collections.defaultdict(set)
        for c in self.cards.values():
            for u in c["cases"]:
                self.case_index[u].add(c["id"])
        self.blocks = collections.defaultdict(list)   # (type, court) -> real ids, oldest first
        for c in sorted(self.cards.values(), key=lambda c: age_key(c["id"])):
            for ct in sorted(c["courts"]):
                self.blocks[(c["type"], ct)].append(c["id"])
        self.tok_index = collections.defaultdict(lambda: collections.defaultdict(set))
        for (t, ct), ids in self.blocks.items():
            for i in ids:
                for tk in self.cards[i]["toks"]:
                    for w in tk:
                        self.tok_index[(t, ct)][w].add(i)
        self.div_cases = collections.defaultdict(set)
        for u in self.case_index:
            self.div_cases[case_parts(u, pivot)[1]].add(u)
        self._co = {}
        self._fit = {}
        self.vec = None
        self.rows = {}

    def _finish(self, c, pivot):
        c["toks"] = [self.nm.core(n) for n in c["names"]][:12]
        c["mtoks"] = self.nm.core(c["main"])
        # the fullest name of the ID (most words, then the main name, then text order): name-structure features use it,
        # so a short variant (a first name Layer A attached) does not make every namesake look like a subset
        c["full"] = max(c["toks"], key=lambda t: (len(t), t == c["mtoks"], " ".join(t)))
        c["sfx"] = set().union(*[self.nm.suffix(n) for n in c["names"]])
        parts = [case_parts(u, pivot) for u in c["cases"]]
        c["courts"] = {p[0] for p in parts}
        c["divs"] = {p[1] for p in parts}
        c["years"] = sorted(p[2] for p in parts if p[2])
        # parties: person vs organisation as Layer A's name gate decided it (majority over the ID's mentions)
        c["is_person"] = c["type"] in ("judge", "counsel") or (c["type"] == "party" and c["org"][1] <= c["org"][0])
        for k in ("rel_counsel", "rel_domains", "rel_phones", "rel_addresses"):
            c.setdefault(k, set())
        if "org_core" not in c:
            op = N.org_parts(c["canonical"])
            c["org_core"], c["org_office"], c["org_loose"] = op["core"], op["office"], op["loose"]
            c["raw_all"] = [c["canonical"]]

    def add_synthetic(self, card):
        self._finish(card, self.cfg["case_numbers"]["year_pivot"])
        src = self.cards[card["source"]]
        card["is_person"] = src["is_person"]   # a fragment is the same kind of name as its source
        if card.get("kind") != "other_generation":
            card["rel_counsel"] = set(src["rel_counsel"])
        self.cards[card["id"]] = card

    def co(self, i):
        if i not in self._co:
            s = set()
            for u in self.cards[i]["cases"]:
                s |= self.case_index.get(u, set())
            s.discard(i)
            self._co[i] = s
        return self._co[i]

    def fitset(self, i, court):
        """Real IDs of the same type in this court whose names the name of i fits."""
        k = (i, court)
        if k not in self._fit:
            c = self.cards[i]
            idx = self.tok_index.get((c["type"], court), {})
            pool = set()
            for ts in c["toks"]:
                if not ts:
                    continue
                last = idx.get(ts[-1], set())
                if len(ts) > 1 and len(ts[0]) >= 2:
                    first = idx.get(ts[0], set())
                    pool |= (first & last) if len(ts[-1]) > 1 else first
                elif len(ts) == 1:
                    pool |= last | idx.get(ts[0], set())
                else:
                    pool |= last
            out = set()
            for j in pool:
                if j == i:
                    continue
                cj = self.cards[j]
                if any(fits_names(ts, tc) for ts in c["toks"] for tc in cj["toks"]):
                    out.add(j)
            self._fit[k] = out
        return self._fit[k]

    # -- vectors
    def set_vectorizer(self, vec):
        self.vec = vec
        ids = sorted(self.cards)
        X = vec.transform([self.cards[i]["main"] for i in ids])
        self.rows = {i: X[k] for k, i in enumerate(ids)}
        self.X = X
        self.row_of = {i: k for k, i in enumerate(ids)}

    def cos(self, a, b):
        return float(self.X[self.row_of[a]].multiply(self.X[self.row_of[b]]).sum())


def fit_vectorizer(cfg, names):
    t = cfg["tfidf"]
    vec = TfidfVectorizer(analyzer=t["analyzer"], ngram_range=(t["ngram_min"], t["ngram_max"]), lowercase=True,
                          dtype=np.float64)
    vec.fit(sorted(set(names)))
    return vec


# ---------------------------------------------------------------- candidates
def candidate_pairs(world, cut, ids_filter=None):
    """Same type AND (share a case OR same court with cosine >= cut). Synthetic-synthetic pairs are skipped.
    Returns {(a, b): court} with a < b."""
    chunk = world.cfg["candidates"]["chunk_rows"]
    blocks = collections.defaultdict(list)
    for i, c in world.cards.items():
        for ct in c["courts"]:
            blocks[(c["type"], ct)].append(i)
    out = {}
    for (t, ct) in sorted(blocks):
        ids = sorted(blocks[(t, ct)])
        if len(ids) < 2:
            continue
        X = world.X[[world.row_of[i] for i in ids]]
        cases = sorted({u for i in ids for u in world.cards[i]["cases"] if u.startswith(ct + ";;")})
        cidx = {u: k for k, u in enumerate(cases)}
        r, cc = [], []
        for k, i in enumerate(ids):
            for u in world.cards[i]["cases"]:
                if u in cidx:
                    r.append(k)
                    cc.append(cidx[u])
        M = sparse.csr_matrix((np.ones(len(r)), (r, cc)), shape=(len(ids), max(1, len(cases))))
        share = (M @ M.T).tocoo()
        found = set()
        for a, b in zip(share.row, share.col):
            if a < b:
                found.add((int(a), int(b)))
        if len(ids) <= world.cfg["candidates"]["dense_max_block"]:
            XT = X.T.tocsc()
            for s in range(0, len(ids), chunk):
                S = (X[s:s + chunk] @ XT).toarray()
                aa, bb = np.nonzero(S >= cut)
                for a, b in zip(aa, bb):
                    a = int(a) + s
                    if a < b:
                        found.add((a, int(b)))
        else:
            # large block: pairs that share a rare word or a rare pair of words, then the exact cosine
            maxdf = world.cfg["candidates"]["max_key_df"]
            keys = collections.defaultdict(list)
            for k, i in enumerate(ids):
                toks = sorted({w for t in world.cards[i]["toks"][:4] for w in t})
                for w in toks:
                    keys[w].append(k)
                for x in range(len(toks)):
                    for y in range(x + 1, min(len(toks), x + 6)):
                        keys[toks[x] + "|" + toks[y]].append(k)
            pc = set()
            for lst in keys.values():
                if 1 < len(lst) <= maxdf:
                    lst = sorted(set(lst))
                    for x in range(len(lst)):
                        for y in range(x + 1, len(lst)):
                            pc.add((lst[x], lst[y]))
            pc = sorted(pc - found)
            for s0 in range(0, len(pc), 200000):
                part = pc[s0:s0 + 200000]
                aa = np.array([a for a, _ in part])
                bb = np.array([b for _, b in part])
                sims = np.asarray(X[aa].multiply(X[bb]).sum(axis=1)).ravel()
                for (a, b), v in zip(part, sims):
                    if v >= cut:
                        found.add((a, b))
        for a, b in found:
            ia, ib = ids[a], ids[b]
            if world.cards[ia]["synthetic"] and world.cards[ib]["synthetic"]:
                continue
            if ids_filter is not None and ia not in ids_filter and ib not in ids_filter:
                continue
            key = (ia, ib) if ia < ib else (ib, ia)
            if key not in out or ct < out[key]:
                out[key] = ct
    return out


# ---------------------------------------------------------------- features
def jacc(a, b):
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def pair_features(world, a, b, court):
    A, B = world.cards[a], world.cards[b]
    # name features over variant pairs (best value of each)
    jw = 0.0
    for ta in A["toks"]:
        for tb in B["toks"]:
            if ta and tb:
                jw = max(jw, JaroWinkler.normalized_similarity(" ".join(ta), " ".join(tb)))
    ta, tb = A["full"], B["full"]
    s, l = (ta, tb) if (len(ta), ta) <= (len(tb), tb) else (tb, ta)
    prefix = 1.0 if s and l[:len(s)] == s else 0.0
    w2i = fit_initials(s, l)
    subset = 1.0 if ordered_fit(s, l, strict=True) else 0.0
    first_c = (1.0 if ta[0] == tb[0] else 0.75 if compat(ta[0], tb[0]) else 0.0) if ta and tb else 0.0
    sur = (1.0 if ta[-1] == tb[-1] else 0.0) if len(ta) >= 2 and len(tb) >= 2 else 0.5
    sfx = 1.0 if A["sfx"] and B["sfx"] and A["sfx"] != B["sfx"] else 0.0
    na, nb = len(A["mtoks"]), len(B["mtoks"])
    shared = A["cases"] & B["cases"]
    coa, cob = world.co(a) - {b}, world.co(b) - {a}
    if A["years"] and B["years"]:
        lo = max(A["years"][0], B["years"][0])
        hi = min(A["years"][-1], B["years"][-1])
        overlap = 1.0 if lo <= hi else 0.0
        gap = float(max(0, lo - hi))
    else:
        overlap, gap = 0.5, 0.0
    if A["type"] == "judge" and A["titles"] and B["titles"]:
        title = 1.0 if A["titles"] & B["titles"] else 0.0
    else:
        title = 0.5
    if A["sides"] and B["sides"]:
        keys = sorted(set(A["sides"]) | set(B["sides"]))
        va = np.array([A["sides"][k] for k in keys], float)
        vb = np.array([B["sides"][k] for k in keys], float)
        side = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
    else:
        side = 0.5
    # competing candidates: other IDs that also fit the shorter name (real IDs only, never the pair itself)
    short = a if (na, -id_num(a)) < (nb, -id_num(b)) or (na == nb and a > b) else b
    fs = world.fitset(short, court) - {a, b}
    sdivs = world.cards[short]["divs"]
    comp_div = sum(1 for j in fs if world.cards[j]["divs"] & sdivs)
    t = A["type"]
    org = 0.0 if (A["is_person"] or B["is_person"]) else 1.0
    if org:
        ca, cb = A["org_core"], B["org_core"]
        same_core = 1.0 if ca and ca == cb else 0.0
        same_loose = 1.0 if A["org_loose"] and A["org_loose"] == B["org_loose"] else 0.0
        office_diff = 1.0 if same_core and A["org_office"] != B["org_office"] else 0.0
        wa, wb = ca.split(), cb.split()
        sh, lo = (wa, wb) if len(wa) <= len(wb) else (wb, wa)
        core_prefix = 1.0 if sh and lo[:len(sh)] == sh and len(sh) < len(lo) else 0.0
        orgf = [org, same_core, same_loose, office_diff, core_prefix,
                math.log1p(len(A["rel_counsel"] & B["rel_counsel"])),
                1.0 if A["rel_domains"] & B["rel_domains"] else 0.0,
                1.0 if A["rel_phones"] & B["rel_phones"] else 0.0,
                1.0 if A["rel_addresses"] & B["rel_addresses"] else 0.0]
    else:
        orgf = [0.0] * 9
    return [
        world.cos(a, b), jw, prefix, subset, first_c, sur, float(w2i or 0), sfx,
        1.0 if min(na, nb) == 1 else 0.0, 1.0 if max(na, nb) == 1 else 0.0, float(min(na, nb)), float(max(na, nb)),
        jacc(A["cases"], B["cases"]), math.log1p(len(shared)),
        1.0 if A["courts"] & B["courts"] else 0.0, 1.0 if A["divs"] & B["divs"] else 0.0,
        jacc(coa, cob), math.log1p(len(coa & cob)),
        overlap, gap, title, side,
        min(A["n"], B["n"]) / max(1, A["n"], B["n"]), math.log1p(min(A["n"], B["n"])),
        float(min(len(fs), 10)), float(min(comp_div, 10)),
        float(t == "judge"), float(t == "party"), float(t == "counsel"), float(t == "firm"),
        1.0 if (A["is_person"] and B["is_person"]) else 0.0,
    ] + orgf


def feature_matrix(world, pairs, log=None):
    """pairs: list of (a, b, court) -> float array (n, len(FEATURES))."""
    X = np.zeros((len(pairs), len(FEATURES)))
    t0 = time.time()
    for k, (a, b, ct) in enumerate(pairs):
        X[k] = pair_features(world, a, b, ct)
        if log and (k + 1) % 200000 == 0:
            log(f"  features: {k + 1}/{len(pairs)} pairs | {time.time() - t0:.0f}s")
    return X


def evidence_group(world, a, b):
    """Organisation pairs are thresholded per evidence group: shared counsel, shared contact details, or name only."""
    A, B = world.cards[a], world.cards[b]
    if A["rel_counsel"] & B["rel_counsel"]:
        return "counsel"
    if (A["rel_domains"] & B["rel_domains"]) or (A["rel_phones"] & B["rel_phones"]) or (A["rel_addresses"] & B["rel_addresses"]):
        return "contact"
    return "name"


def threshold_key(world, a, b):
    kd = pair_kind(world, a, b)
    return f"{kd}|{evidence_group(world, a, b)}" if kd in ORG_KINDS else kd


# ---------------------------------------------------------------- reference labels
def load_fjc(cfg):
    p = cfg_path(cfg, cfg["reference"]["fjc_list"])
    by_first = collections.defaultdict(list)
    if not p.exists():
        return by_first
    for r in csv.DictReader(open(p, encoding="utf-8", errors="replace")):
        first = norm_text(r.get("First Name")).split()
        toks = first + norm_text(r.get("Middle Name")).split() + norm_text(r.get("Last Name")).split()
        if first and len(toks) >= 2:
            by_first[first[0]].append((r.get("nid") or r.get("jid"), toks))
    return by_first


def fjc_ids(world, fjc):
    """Judge ID -> FJC nid when its name (2+ words, full first name) fits exactly one FJC judge, for every variant
    that fits exactly one; variants that point to different FJC judges leave the ID unmapped."""
    out = {}
    if not fjc:
        return out
    for i, c in world.cards.items():
        if c["type"] != "judge":
            continue
        got = set()
        for ts in c["toks"]:
            if len(ts) < 2 or len(ts[0]) < 2:
                continue
            hits = {nid for nid, ft in fjc.get(ts[0], []) if fits_names(ts, ft)}
            if len(hits) == 1:
                got |= hits
        if len(got) == 1:
            out[i] = next(iter(got))
    return out


def key_gold(world, mentions, key_dir):
    """Entity ID -> gold ID of the pilot answer key (score.py matcher), when its key-file mentions match exactly one."""
    sys.path.insert(0, str(HERE))
    import score as S
    key = S.load_key(key_dir)
    files = set(open(Path(key_dir) / "work/sample200.txt").read().split())
    match = S.build_matcher(key)
    golds = collections.defaultdict(set)
    for m in mentions:
        t = m.get("entity_type")
        if t not in ("judge", "party", "firm") or m.get("entity_id") not in world.cards:
            continue
        f = S.ucid_to_file(m["ucid"])
        if f not in files:
            continue
        g = match(t, f, m.get("raw_name") or "")
        if g:
            golds[m["entity_id"]].add(g)
    return {i: next(iter(g)) for i, g in golds.items() if len(g) == 1}, sum(1 for g in golds.values() if len(g) > 1)


# ---------------------------------------------------------------- fragments
def make_fragments(world, cfg):
    """Synthetic IDs cut from confident full names and placed in the same cases (label source c)."""
    fc = cfg["fragments"]
    rng = random.Random(fc["seed"])
    made = []
    for t in TYPES:
        src = sorted(i for i, c in world.cards.items() if c["type"] == t and c["is_person"] and not c["synthetic"]
                     and len(c["full"]) >= 2 and len(c["full"][0]) >= 2 and c["cases"])
        if len(src) > fc["max_sources_per_type"]:
            src = sorted(rng.sample(src, fc["max_sources_per_type"]))
        for i in src:
            c = world.cards[i]
            toks = c["full"]
            kinds = []
            for k in fc["kinds"]:
                if k == "first_only":
                    kinds.append((k, [toks[0]]))
                elif k == "initial_surname":
                    kinds.append((k, [toks[0][0], toks[-1]]))
                elif k == "with_title" and t == "judge" and c["titles"]:
                    # "<title> <surname>" as Layer A stores it: titles are cut from the name into the titles field
                    kinds.append((k, [toks[-1]]))
                elif k == "first_words" and len(toks) >= 3 and len(toks[-2]) >= 2:
                    kinds.append((k, toks[:-1]))
                elif k == "other_generation" and c["sfx"]:
                    # a relative: the same name with a different generation suffix, in the same cases (a negative)
                    other = sorted(set(world.nm.suffixes) - c["sfx"])
                    kinds.append((k, toks + [rng.choice(other)]))
            rng.shuffle(kinds)
            for kind, ft in kinds[:fc["per_source"]]:
                own = sorted(c["cases"])
                k_in = rng.randint(fc["cases_in"][0], min(fc["cases_in"][1], len(own)))
                cases = set(rng.sample(own, k_in))
                if t == "judge" and kind != "other_generation":
                    div = sorted(c["divs"])[0]
                    others = sorted(world.div_cases.get(div, set()) - c["cases"])
                    k_out = rng.randint(fc["cases_out"][0], min(fc["cases_out"][1], len(others)))
                    cases |= set(rng.sample(others, k_out))
                if not cases:
                    cases = {rng.choice(own)}
                fid = f"frag_{t}_{len(made):06d}"
                n = len(cases) * rng.randint(1, 2)
                sides = collections.Counter()
                if c["sides"]:
                    top = c["sides"].most_common(1)[0][0]
                    sides[top] = n
                name = " ".join(ft)
                card = {"id": fid, "type": t, "canonical": name, "main": name, "names": [name], "cases": cases,
                        "n": n, "titles": set(c["titles"]) if kind == "with_title" else set(), "sides": sides,
                        "roles": collections.Counter(), "org": [1, 0], "raw": [name], "synthetic": True,
                        "source": i, "kind": kind}
                world.add_synthetic(card)
                made.append(fid)
    return made


# ---------------------------------------------------------------- training
def build_training(world, mentions, cfg, key_dir, fjc):
    """Returns pairs [(a, b, court)], labels, label source per pair, and counts."""
    gold, impure = (key_gold(world, mentions, key_dir) if key_dir and Path(key_dir).exists() else ({}, 0))
    fj = fjc_ids(world, fjc)
    frags = make_fragments(world, cfg)
    names = [c["main"] for c in world.cards.values()]
    vec = fit_vectorizer(cfg, names)
    world.set_vectorizer(vec)
    # data-derived candidate cut: low quantile of the cosine over labelled same-court positives
    pos_cos = []
    by_label = collections.defaultdict(list)
    for i, g in gold.items():
        by_label[("key", g)].append(i)
    for i, g in fj.items():
        by_label[("fjc", g)].append(i)
    for ids in by_label.values():
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                a, b = ids[x], ids[y]
                if world.cards[a]["type"] == world.cards[b]["type"] and world.cards[a]["courts"] & world.cards[b]["courts"]:
                    pos_cos.append(world.cos(a, b))
    for f in frags:
        if world.cards[f]["kind"] != "other_generation":
            pos_cos.append(world.cos(f, world.cards[f]["source"]))
    cut = float(np.quantile(pos_cos, cfg["candidates"]["positive_quantile"])) if pos_cos else 0.3
    cands = candidate_pairs(world, cut)
    # filter recall: share of labelled same-court positives that the candidate filter keeps
    kept = tot = 0
    for ids in by_label.values():
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                a, b = sorted((ids[x], ids[y]))
                if world.cards[a]["type"] == world.cards[b]["type"] and world.cards[a]["courts"] & world.cards[b]["courts"]:
                    tot += 1
                    kept += (a, b) in cands
    pairs, y, src = [], [], []
    for (a, b), ct in sorted(cands.items()):
        A, B = world.cards[a], world.cards[b]
        if A["synthetic"] or B["synthetic"]:
            f, o = (a, b) if A["synthetic"] else (b, a)
            F, O = world.cards[f], world.cards[o]
            s = F["source"]
            if F["kind"] == "other_generation":
                lab = 0
            elif o == s:
                lab = 1
            else:
                # similar real names elsewhere are negatives (label source c). Unknown, so skipped: a name that also
                # fits the source's own name and shares a case with the fragment, or (judges, who sit on many cases
                # of a division) any judge whose name fits the source's name - it may be an unmerged copy of the source
                fits_src = any(fits_names(t1, t2) or fits_names(t2, t1) for t1 in O["toks"] for t2 in world.cards[s]["toks"])
                if fits_src and (F["cases"] & O["cases"] or F["type"] == "judge"):
                    continue
                lab = 0
            pairs.append((a, b, ct)); y.append(lab); src.append("fragment")
        elif a in gold and b in gold:
            pairs.append((a, b, ct)); y.append(int(gold[a] == gold[b])); src.append("answer_key")
        elif a in fj and b in fj:
            pairs.append((a, b, ct)); y.append(int(fj[a] == fj[b])); src.append("fjc")
    counts = collections.Counter()
    for (a, b, ct), lab, s in zip(pairs, y, src):
        counts[(s, world.cards[a]["type"], lab)] += 1
    info = {"gold_mapped_ids": len(gold), "gold_impure_ids": impure, "fjc_mapped_ids": len(fj), "fragments": len(frags),
            "candidate_cut": cut, "positive_quantile": cfg["candidates"]["positive_quantile"], "candidate_pairs": len(cands), "filter_recall_labelled": [kept, tot]}
    return vec, cut, pairs, np.array(y), np.array(src), counts, info, fj


def make_models(cfg):
    m = cfg["model"]
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=m["lr_C"], max_iter=5000))
    hgb = HistGradientBoostingClassifier(max_iter=m["hgb_max_iter"], learning_rate=m["hgb_learning_rate"],
                                         max_leaf_nodes=m["hgb_max_leaf_nodes"], l2_regularization=m["hgb_l2"],
                                         early_stopping=False, random_state=m["random_state"])
    return {"logistic_regression": lr, "hist_gradient_boosting": hgb}


def calibrated(est, X, y, groups, cfg):
    k = min(cfg["model"]["inner_folds"], len(set(groups)))
    splits = list(GroupKFold(n_splits=k).split(X, y, groups))
    cal = CalibratedClassifierCV(est, method=cfg["model"]["calibration"], cv=splits)
    cal.fit(X, y)
    return cal


def threshold_high(y, p, target, min_support):
    """Smallest t such that precision(p >= t) >= target for every threshold from the top down to t
    (thresholds with fewer than min_support pairs above them are skipped). None if never reached."""
    order = np.argsort(-p, kind="stable")
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys)
    best = None
    k = 0
    n = len(ps)
    while k < n:
        j = k
        while j + 1 < n and ps[j + 1] == ps[k]:
            j += 1
        cnt = j + 1
        prec = tp[j] / cnt
        if cnt >= min_support:
            if prec >= target:
                best = float(ps[k])
            else:
                break
        k = j + 1
    return best


def threshold_real(y, p, start, target):
    """Smallest t >= start such that the pairs at or above every threshold from the top down to t keep precision >=
    target (an empty set passes). Used on the real labelled pairs (answer key, FJC) of one kind. None if impossible."""
    if start is None:
        return None
    order = np.argsort(-p, kind="stable")
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys)
    worst = None
    k, n = 0, len(ps)
    while k < n:
        j = k
        while j + 1 < n and ps[j + 1] == ps[k]:
            j += 1
        if tp[j] / (j + 1) < target and worst is None:
            worst = ps[k]          # the highest failing threshold: t must be above it
        k = j + 1
    if worst is None or worst < start:
        return start
    return None if worst >= 1.0 else float(np.nextafter(worst, 2.0))


def threshold_low(y, p, recall):
    pos = np.sort(p[y == 1])
    if not len(pos):
        return 0.5
    return float(pos[int(math.floor((1 - recall) * len(pos)))])


def train(world, mentions, cfg, key_dir, log):
    fjc = load_fjc(cfg)
    vec, cut, pairs, y, src, counts, info, fj = build_training(world, mentions, cfg, key_dir, fjc)
    log(f"training pairs: {len(pairs)} (positives {int(y.sum())}); candidate cut {cut:.3f}")
    X = feature_matrix(world, pairs)
    groups = np.array([ct for _, _, ct in pairs])
    k = min(cfg["model"]["outer_folds"], len(set(groups)))
    outer = list(GroupKFold(n_splits=k).split(X, y, groups))
    court_fold = {}
    for f, (_, te) in enumerate(outer):
        for g in set(groups[te]):
            court_fold[g] = f
    oof = {}
    fold_models = {}
    for name, est in make_models(cfg).items():
        p = np.zeros(len(y))
        fold_models[name] = []
        for f, (tr, te) in enumerate(outer):
            cal = calibrated(est, X[tr], y[tr], groups[tr], cfg)
            p[te] = cal.predict_proba(X[te])[:, 1]
            fold_models[name].append(cal)
        oof[name] = p
        log(f"  {name}: held-out average precision {average_precision_score(y, p):.4f}")
    metrics = {}
    for name, p in oof.items():
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        metrics[name] = {"average_precision": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)),
                         "log_loss": float(log_loss(y, pc)),
                         "by_source_average_precision": {s: float(average_precision_score(y[src == s], p[src == s]))
                                                         for s in sorted(set(src)) if 0 < y[src == s].sum() < (src == s).sum()}}
    chosen = max(metrics, key=lambda n: (metrics[n][cfg["model"]["pick_by"]], n))
    p = oof[chosen]
    th = cfg["thresholds"]
    # T_high per pair kind (organisations: per kind AND evidence group): >= precision_target on held-out courts;
    # a group that never reaches it gets no auto-merges
    by_kind = {}
    kinds = np.array([str(threshold_key(world, a, b)) for a, b, _ in pairs])
    for kd in sorted(set(kinds)):
        m_ = kinds == kd
        t_all = threshold_high(y[m_], p[m_], th["precision_target"], th["min_support"]) if y[m_].sum() >= th["min_support"] else None
        r_ = m_ & (src != "fragment")
        by_kind[kd] = threshold_real(y[r_], p[r_], t_all, th["precision_target"]) if r_.any() else t_all
    t_low = threshold_low(y, p, th["low_recall"])
    t_of = np.array([by_kind[kd] if by_kind[kd] is not None else np.inf for kd in kinds])
    at = p >= t_of
    held = {}
    real = src != "fragment"
    for name_, m_ in [("all kinds", np.ones(len(y), bool))] + [(kd, kinds == kd) for kd in sorted(set(kinds))] + \
            [(f"{kd}, real labels only", (kinds == kd) & real) for kd in sorted(set(kinds[real]))] + \
            [(f"source: {s}", src == s) for s in sorted(set(src))]:
        a_ = at & m_
        held[name_] = {"precision": float(y[a_].mean()) if a_.any() else None, "recall": float(y[a_].sum() / max(1, y[m_].sum())),
                       "n_above": int(a_.sum()), "positives": int(y[m_].sum()), "pairs": int(m_.sum())}
    log(f"chosen: {chosen}; T_high by kind {by_kind}; T_low {t_low:.4f}")
    final = calibrated(make_models(cfg)[chosen], X, y, groups, cfg)
    # explanations: LR coefficients (standardised), HGB permutation importances
    lr = make_models(cfg)["logistic_regression"].fit(X, y)
    coefs = dict(zip(FEATURES, map(float, lr[-1].coef_[0])))
    hgb = make_models(cfg)["hist_gradient_boosting"].fit(X, y)
    rs = np.random.RandomState(cfg["model"]["random_state"])
    sub = rs.choice(len(y), size=min(len(y), 20000), replace=False)
    imp = permutation_importance(hgb, X[sub], y[sub], scoring="average_precision", n_repeats=5,
                                 random_state=cfg["model"]["random_state"])
    importances = dict(zip(FEATURES, map(float, imp.importances_mean)))
    medians = np.median(X, axis=0)
    # LLM evaluation pairs: real labelled pairs (answer key / FJC) from held-out courts first, then fragments
    rng = random.Random(cfg["fragments"]["seed"])
    n_eval = cfg["llm"]["n_eval_pairs"]
    eval_pairs = []
    for want in (1, 0):
        pool = [k for k in range(len(y)) if y[k] == want]
        pool.sort(key=lambda k: (src[k] == "fragment", court_fold.get(groups[k], 0), pairs[k][0], pairs[k][1]))
        eval_pairs += pool[:n_eval // 2]
    eval_cards = [{"a": card_view(world, pairs[k][0]), "b": card_view(world, pairs[k][1]), "label": int(y[k]),
                   "source": str(src[k]), "court": pairs[k][2],
                   "competing": competing_view(world, pairs[k][0], pairs[k][1], pairs[k][2])} for k in eval_pairs]
    bundle = {
        "version": MODEL_VERSION, "features": FEATURES, "vectorizer": vec, "candidate_cut": cut, "model": final,
        "model_name": chosen, "fold_models": fold_models[chosen], "court_fold": court_fold,
        "thresholds": {"T_high_by_kind": by_kind, "T_low": t_low, **{k: th[k] for k in th if k != "source"}},
        "metrics": metrics, "held_out_at_T_high": held, "lr_coefficients": coefs, "hgb_importances": importances,
        "medians": medians.tolist(), "train_counts": {f"{s}|{t}|{l}": n for (s, t, l), n in sorted(counts.items())},
        "train_info": info, "llm_eval": eval_cards, "trained_on": Path(world.cfg.get("_input", "")).name,
    }
    return bundle, (pairs, X, y, src, groups, oof)


# ---------------------------------------------------------------- consolidation
def card_view(world, i, snippets=None):
    c = world.cards[i]
    co_names = sorted({world.cards[j]["canonical"] for j in world.co(i) if j in world.cards and not world.cards[j]["synthetic"]})
    v = {"id": i, "type": c["type"], "canonical": c["canonical"], "names": c["names"][:10], "raw_names": c["raw"],
         "courts": sorted(c["courts"]), "divisions": sorted(c["divs"]), "n_cases": len(c["cases"]),
         "cases": sorted(c["cases"])[:10], "years": [c["years"][0], c["years"][-1]] if c["years"] else [],
         "titles": sorted(c["titles"]), "sides": dict(c["sides"]), "roles": dict(c["roles"]), "n_mentions": c["n"],
         "co_occurring": co_names[:10], "n_co_occurring": len(co_names)}
    if snippets is not None:
        v["snippets"] = snippets.get(i, [])
    return v


def competing_view(world, a, b, court):
    na, nb = len(world.cards[a]["mtoks"]), len(world.cards[b]["mtoks"])
    short = a if (na, -id_num(a)) < (nb, -id_num(b)) or (na == nb and a > b) else b
    fs = world.fitset(short, court) - {a, b}
    return sorted(world.cards[j]["canonical"] for j in fs if not world.cards[j]["synthetic"])[:10]


class Clusters:
    def __init__(self, world, fjc_map):
        self.w = world
        self.parent = {}
        self.members = {}
        self.fjc = fjc_map

    def root(self, i):
        while self.parent.get(i, i) != i:
            i = self.parent[i]
        return i

    def mem(self, r):
        return self.members.get(r, [r])

    def refuse(self, ra, rb, pair_scores, t_low):
        A, B = self.mem(ra), self.mem(rb)
        cards = self.w.cards
        if len({cards[i]["type"] for i in A + B if i in cards}) > 1:
            return "would mix entity types"
        fj = {self.fjc[i] for i in A + B if i in self.fjc}
        if len(fj) > 1:
            return "would join two different FJC judges"
        fa, fb = collections.defaultdict(set), collections.defaultdict(set)
        for i in A:
            if i in cards:
                for s, f in full_firsts(cards[i]).items():
                    fa[s] |= f
        for i in B:
            if i in cards:
                for s, f in full_firsts(cards[i]).items():
                    fb[s] |= f
        for s in set(fa) & set(fb):
            if fa[s] - fb[s] and fb[s] - fa[s]:
                return "would join two different full first names with the same surname"
        if all(i in cards and cards[i]["type"] == "party" and not cards[i]["is_person"] for i in A + B):
            ca = set().union(*[cards[i]["cases"] for i in A])
            if any(ca & cards[j]["cases"] for j in B):
                return "organisations listed as separate parties in one case"
        # the model's own keep-apart decisions between the two groups are respected (no chaining through them)
        for x in A:
            for y in B:
                k = pair_key(x, y)
                if k in pair_scores and pair_scores[k] < t_low:
                    return "the model keeps another pair between these groups apart"
        return None

    def union(self, ra, rb):
        keep, lose = sorted((ra, rb), key=age_key)
        self.parent[lose] = keep
        self.members[keep] = self.mem(keep) + self.mem(lose)
        self.members.pop(lose, None)
        return keep, lose


def pair_kind(world, a, b):
    """Threshold group of a pair: the entity type, with parties split into person and organisation pairs."""
    t = world.cards[a]["type"]
    if t == "party":
        pa, pb = world.cards[a]["is_person"], world.cards[b]["is_person"]
        return "party_person" if pa and pb else "party_org" if not (pa or pb) else "party_person_org"
    return t


def pair_key(a, b):
    return f"{a}|{b}" if a < b else f"{b}|{a}"


def explain(bundle, xs, ps):
    """Top 3 reasons per pair: features whose replacement by the training median lowers p the most."""
    if not len(xs):
        return []
    medv = np.array(bundle["medians"])
    F = len(FEATURES)
    rows = np.repeat(xs, F, axis=0)
    for f in range(F):
        rows[f::F, f] = medv[f]
    pp = bundle["model"].predict_proba(rows)[:, 1].reshape(len(xs), F)
    out = []
    for k in range(len(xs)):
        d = ps[k] - pp[k]
        top = np.argsort(-d, kind="stable")[:3]
        out.append([{"feature": FEATURES[f], "value": round(float(xs[k][f]), 4), "effect": round(float(d[f]), 4)} for f in top])
    return out


def consolidate(world, bundle, pair_list, probs, state, fjc_map, log, run_no, llm=None):
    """Apply old redirects, then new merges in descending score. Returns new redirect rows, review rows, cluster."""
    th = bundle["thresholds"]
    t_low = th["T_low"]
    t_by = {k: (v if v is not None else math.inf) for k, v in th["T_high_by_kind"].items()}
    cl = Clusters(world, fjc_map)
    for r in state["redirects"]:
        a, b = cl.root(r["old_id"]), cl.root(r["surviving_id"])
        if a != b:
            cl.parent[a] = b
            cl.members[b] = cl.mem(b) + cl.mem(a)
            cl.members.pop(a, None)
    scores = state["pair_scores"]
    order = sorted(range(len(pair_list)), key=lambda k: (-probs[k], pair_list[k][0], pair_list[k][1]))
    merges, review = [], []
    for k in order:
        a, b, ct = pair_list[k]
        p = probs[k]
        ra, rb = cl.root(a), cl.root(b)
        if ra == rb:
            continue
        t_high = t_by.get(threshold_key(world, a, b), math.inf)
        if p >= t_high:
            why = cl.refuse(ra, rb, scores, t_low)
            if why:
                review.append((k, f"refused: {why}"))
                continue
            keep, lose = cl.union(ra, rb)
            merges.append({"old_id": lose, "surviving_id": keep, "score": round(float(p), 6), "via_pair": [a, b],
                           "run": run_no, "decided_by": bundle["model_name"]})
        elif p >= t_low:
            review.append((k, "uncertain"))
    return merges, review, cl


def related_links(world, pair_list, probs, final_of, bundle):
    """Organisation pairs that stay apart but are linked: RELATED = an office of the same organisation (same core name,
    different office) or a parent / subsidiary (one core name is the first words of the other, with shared counsel or
    contact details, or a score in the review band). Both IDs are kept."""
    t_low = bundle["thresholds"]["T_low"]
    out = {}
    for k, (a, b, ct) in enumerate(pair_list):
        A, B = world.cards[a], world.cards[b]
        if A["is_person"] or B["is_person"] or A["type"] not in ("party", "firm"):
            continue
        fa, fb = final_of.get(a, a), final_of.get(b, b)
        if fa == fb:
            continue
        ev = evidence_group(world, a, b)
        rel = None
        if A["org_core"] and A["org_core"] == B["org_core"] and A["org_office"] != B["org_office"]:
            rel = "office"
        else:
            wa, wb = A["org_core"].split(), B["org_core"].split()
            sh, lo = (wa, wb) if len(wa) <= len(wb) else (wb, wa)
            if sh and len(sh) < len(lo) and lo[:len(sh)] == sh and (ev != "name" or probs[k] >= t_low):
                rel = "parent_or_subsidiary"
        if rel:
            key = tuple(sorted((fa, fb)))
            row = {"id_a": key[0], "id_b": key[1], "entity_type": A["type"], "relation": rel, "score": round(float(probs[k]), 6),
                   "evidence": ev, "via_pair": [a, b],
                   "names": [A["canonical"], B["canonical"]] if key[0] == fa else [B["canonical"], A["canonical"]]}
            if key not in out or (row["relation"], -row["score"]) < (out[key]["relation"], -out[key]["score"]):
                out[key] = row
    return [out[k] for k in sorted(out)]


def display_of(world, entity_rows_by_id, members):
    """canonical name of a final entity: the most complete variant over all member IDs (v4_names.display_name)."""
    raws, org = [], False
    for i in members:
        c = world.cards.get(i)
        if c:
            raws += c["raw_all"] or [c["canonical"]]
            org = org or (c["type"] == "firm") or (c["type"] == "party" and not c["is_person"])
        else:
            raws.append(entity_rows_by_id[i].get("canonical_name") or "")
    if members and entity_rows_by_id[members[0]]["entity_type"] == "judge":
        raws = [entity_rows_by_id[i].get("canonical_name") or "" for i in members]
    return N.display_name(raws, N.ORG if org else N.PERSON) or entity_rows_by_id[members[0]].get("canonical_name")


def collapse(redirects):
    to = {}
    for r in redirects:
        to[r["old_id"]] = r["surviving_id"]

    def final(i):
        seen = set()
        while i in to and i not in seen:
            seen.add(i)
            i = to[i]
        return i
    return {i: final(i) for i in to}


# ---------------------------------------------------------------- llm
def llm_prompt(pair):
    a, b = pair["a"], pair["b"]
    t = a["type"]
    return (f"Two records from US federal court dockets were extracted as {t} entities. Decide whether they are the "
            f"same real {t}.\nRecord A: {json.dumps(a, sort_keys=True)}\nRecord B: {json.dumps(b, sort_keys=True)}\n"
            f"Other {t} IDs in the same court whose names also fit the shorter name: {json.dumps(pair.get('competing', []))}\n"
            'Reply with JSON only: {"answer": "same" | "different" | "unsure", "reason": "<one sentence>"}')


class LLM:
    def __init__(self, mode, cfg, out_dir, log):
        self.mode, self.cfg, self.log = mode, cfg["llm"], log
        self.cache_path = Path(out_dir) / "llm_cache.jsonl"
        self.cache = {r["key"]: r for r in read_jsonl(self.cache_path)}
        self.ok = True
        self.calls = 0
        if mode == "local":
            try:
                tags = json.loads(urllib.request.urlopen(self.cfg["local_url"] + "/api/tags", timeout=5).read())
                names = {m["name"] for m in tags.get("models", [])}
                if self.cfg["local_model"] not in names:
                    log(f"LLM off: Ollama model {self.cfg['local_model']} is not pulled (ollama pull {self.cfg['local_model']})")
                    self.ok = False
            except Exception:
                log("LLM off: Ollama is not installed or not running (install from ollama.com, then `ollama serve`)")
                self.ok = False
        elif mode == "api":
            if not os.environ.get(self.cfg["api_key_env"]):
                log(f"LLM off: environment variable {self.cfg['api_key_env']} is not set")
                self.ok = False

    def ask(self, pair):
        prompt = llm_prompt(pair)
        model = self.cfg["local_model"] if self.mode == "local" else self.cfg["api_model"]
        key = hashlib.sha256(f"{self.mode}|{model}|{prompt}".encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        text = ""
        try:
            if self.mode == "local":
                body = {"model": model, "prompt": prompt, "stream": False, "format": "json",
                        "options": {"temperature": 0, "seed": 0}}
                req = urllib.request.Request(self.cfg["local_url"] + "/api/generate", json.dumps(body).encode(),
                                             {"Content-Type": "application/json"})
                text = json.loads(urllib.request.urlopen(req, timeout=self.cfg["timeout_s"]).read())["response"]
            else:
                body = {"model": model, "max_tokens": 200, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
                req = urllib.request.Request(self.cfg["api_url"], json.dumps(body).encode(),
                                             {"Content-Type": "application/json", "anthropic-version": "2023-06-01",
                                              "x-api-key": os.environ[self.cfg["api_key_env"]]})
                text = json.loads(urllib.request.urlopen(req, timeout=self.cfg["timeout_s"]).read())["content"][0]["text"]
            m = re.search(r"\{.*\}", text, re.S)
            ans = json.loads(m.group(0)) if m else {}
        except Exception as ex:
            ans = {"answer": "unsure", "reason": f"call failed: {type(ex).__name__}"}
        a = str(ans.get("answer", "unsure")).strip().lower()
        row = {"key": key, "model": model, "answer": a if a in ("same", "different", "unsure") else "unsure",
               "reason": str(ans.get("reason", ""))[:300]}
        self.calls += 1
        self.cache[key] = row
        with open(self.cache_path, "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        return row

    def evaluate(self, eval_pairs):
        res = collections.Counter()
        for pr in eval_pairs:
            ans = self.ask(pr)["answer"]
            res[(ans, pr["label"])] += 1
        same = res[("same", 1)] + res[("same", 0)]
        prec = res[("same", 1)] / same if same else None
        return {"n": len(eval_pairs), "precision_same": prec, "same_correct": res[("same", 1)], "same_wrong": res[("same", 0)],
                "different_correct": res[("different", 0)], "different_wrong": res[("different", 1)],
                "unsure": res[("unsure", 0)] + res[("unsure", 1)],
                "positives": sum(1 for p in eval_pairs if p["label"] == 1),
                "sources": dict(collections.Counter(p["source"] for p in eval_pairs))}


def load_snippets(world, case_dirs, ids, cfg):
    """Up to max_snippets docket lines per ID that contain one of its name words (needs --cases)."""
    if not case_dirs:
        return {}
    files = {}
    for d in case_dirs:
        for fp in sorted(Path(d).expanduser().glob("*.json")):
            files.setdefault(fp.name, fp)
    sys.path.insert(0, str(HERE))
    import score as S
    by_case = collections.defaultdict(set)
    for i in ids:
        for u in sorted(world.cards[i]["cases"])[:20]:
            by_case[u].add(i)
    out = collections.defaultdict(list)
    lc = cfg["llm"]
    for u in sorted(by_case):
        fp = files.get(S.ucid_to_file(u))
        if not fp:
            continue
        try:
            dk = json.load(open(fp)).get("docket") or []
        except Exception:
            continue
        for i in sorted(by_case[u]):
            if len(out[i]) >= lc["max_snippets"]:
                continue
            words = {w for t in world.cards[i]["toks"] for w in t if len(w) >= 3}
            for e in dk:
                txt = e.get("docket_text") or ""
                low = set(norm_text(txt).split())
                if words & low:
                    out[i].append(txt[:lc["snippet_chars"]])
                    if len(out[i]) >= lc["max_snippets"]:
                        break
    return out


# ---------------------------------------------------------------- scoring (pilot answer key)
def key_scores(mentions, final_of, key_dir):
    sys.path.insert(0, str(HERE))
    import score as S
    key = S.load_key(key_dir)
    files = set(open(Path(key_dir) / "work/sample200.txt").read().split())
    match = S.build_matcher(key)
    out = {}
    for label, mp in (("before", {}), ("after", final_of)):
        rows = []
        for m in mentions:
            t = m.get("entity_type")
            if t not in ("judge", "party", "firm"):
                continue
            sid = f"noid:{m['mention_id']}" if m.get("unresolved") else mp.get(m["entity_id"], m["entity_id"])
            rows.append((t, S.ucid_to_file(m["ucid"]), m.get("raw_name") or "", f"LB:{sid}"))
        for t in ("judge", "party", "firm"):
            prep = S.prepare(t, rows, key, match, files)
            e, r, c = S.aggregate(prep, {f: 1 for f in files})
            out[(label, t)] = {"precision": r[0], "recall": r[1], "f1": r[2], "extraction_f1": e[2]}
    return out


def pilot_held_out(world, mentions, bundle, key_dir, fjc_map, log):
    """Pilot evaluation with court-held-out fold models: each court is scored by the fold model that never saw it."""
    pairs = sorted((a, b, ct) for (a, b), ct in candidate_pairs(world, bundle["candidate_cut"]).items()
                   if not world.cards[a]["synthetic"] and not world.cards[b]["synthetic"])
    X = feature_matrix(world, pairs)
    nf = len(bundle["fold_models"])
    fold = np.array([bundle["court_fold"].get(ct, zlib.crc32(ct.encode()) % nf) for _, _, ct in pairs])
    p = np.zeros(len(pairs))
    for f in range(nf):
        m = fold == f
        if m.any():
            p[m] = bundle["fold_models"][f].predict_proba(X[m])[:, 1]
    st = {"redirects": [], "pair_scores": {pair_key(a, b): float(q) for (a, b, _), q in zip(pairs, p)}}
    merges, review, cl = consolidate(world, bundle, pairs, p, st, fjc_map, log, 0)
    final_of = collapse(merges)
    sc = key_scores(mentions, final_of, key_dir)
    by_type = collections.Counter(world.cards[r["old_id"]]["type"] for r in merges)
    return {"scores": {f"{l}|{t}": v for (l, t), v in sc.items()}, "merges_by_type": dict(by_type),
            "review_by_type": dict(collections.Counter(world.cards[pairs[k][0]]["type"] for k, _ in review)),
            "pairs_scored": len(pairs)}


# ---------------------------------------------------------------- outputs
def write_model_card(path, bundle):
    th, info = bundle["thresholds"], bundle["train_info"]
    L = ["# Layer B model card", "",
         f"- Model kept: **{bundle['model_name']}** (isotonic-calibrated; the better held-out average precision of the two)",
         f"- Trained on the V4 output folder `{bundle['trained_on']}`; version `{bundle['version']}`",
         f"- Candidate filter (speed only): same type AND (share a case OR same court with char n-gram TF-IDF cosine >= "
         f"{bundle['candidate_cut']:.4f}). The cut is the {100 * info['positive_quantile']:g}th percentile of the cosine over labelled same-court positive pairs. "
         f"The filter keeps {info['filter_recall_labelled'][0]} of {info['filter_recall_labelled'][1]} labelled real same-court positive pairs.",
         "", "## Thresholds (from held-out court folds)", "",
         "- **T_high, per pair kind** (merge if p >= T_high): the smallest score at which held-out precision stays >= "
         f"{th['precision_target']} from the top down, needing >= {th['min_support']} positives of that kind, and at which the "
         "real labelled pairs of that kind (answer key, FJC) also keep that precision (no real negative above it when there "
         "are no real positives). A kind that never reaches it gets no automatic merges (review queue only): "
         + ", ".join(f"{k} {('%.4f' % v) if v is not None else 'not reached'}" for k, v in th["T_high_by_kind"].items()) + ".",
         f"- **T_low = {th['T_low']:.4f}**: review queue for T_low <= p < T_high. {100 * th['low_recall']:.0f}% of held-out true pairs score at or above it.",
         "", "Held-out results at T_high (by kind and by label source):", "", "| group | pairs | positives | pairs >= T_high | precision | recall |", "|---|---|---|---|---|---|"]
    for s, v in bundle["held_out_at_T_high"].items():
        L.append(f"| {s} | {v.get('pairs', '')} | {v.get('positives', '')} | {v['n_above']} | "
                 f"{'' if v['precision'] is None else '%.4f' % v['precision']} | {v['recall']:.4f} |")
    L += ["", "## Both models on held-out courts", "", "| model | average precision | Brier | log loss | AP by source |", "|---|---|---|---|---|"]
    for n, m in bundle["metrics"].items():
        L.append(f"| {n} | {m['average_precision']:.4f} | {m['brier']:.4f} | {m['log_loss']:.4f} | "
                 + ", ".join(f"{s} {v:.3f}" for s, v in m["by_source_average_precision"].items()) + " |")
    L += ["", "## Training pairs by source (label 1 = same, 0 = different)", "", "| source | type | label 1 | label 0 |", "|---|---|---|---|"]
    agg = collections.defaultdict(lambda: [0, 0])
    for k, n in bundle["train_counts"].items():
        s, t, l = k.split("|")
        agg[(s, t)][0 if l == "1" else 1] += n
    for (s, t), (p, n) in sorted(agg.items()):
        L.append(f"| {s} | {t} | {p} | {n} |")
    L += ["", f"- Answer-key IDs mapped to one gold entity: {info['gold_mapped_ids']} (IDs matching 2+ gold entities, skipped: {info['gold_impure_ids']})",
          f"- Judge IDs mapped to exactly one FJC judge: {info['fjc_mapped_ids']}",
          f"- Synthetic fragments: {info['fragments']}; candidate pairs in the training run: {info['candidate_pairs']}",
          "- Split: group k-fold by court; a test court is never seen in training (outer folds for the held-out numbers, "
          "inner folds for calibration).",
          "", "## Features, logistic-regression weights (standardised) and gradient-boosting permutation importance", "",
          "| feature | LR weight | HGB importance (AP drop) |", "|---|---|---|"]
    for f in FEATURES:
        L.append(f"| {f} | {bundle['lr_coefficients'][f]:+.3f} | {bundle['hgb_importances'][f]:.4f} |")
    L += ["", "Feature meanings are in the FEATURES list at the top of v4_consolidate.py.", "",
          "## Fixed constraints (the only hand rules)", "",
          "1. never mix entity types; 2. never put two different FJC judges in one ID; 3. never put two different full first "
          "names with the same surname in one ID; 4. never merge organisations listed as separate parties in one case. "
          "Merges are applied in descending score and also respect the model's own keep-apart scores (p < T_low) between "
          "the two groups, so no merge chains through a pair the model separated. Organisation thresholds are set per "
          "evidence group (shared counsel / shared contact details / name only); a group that never reaches the "
          "precision target merges nothing automatically. Organisation pairs that stay apart are linked as RELATED in "
          "related_links.jsonl: an office (same core name, different office) or a parent / subsidiary (one core name is "
          "the first words of the other, with relational evidence or a score in the review band).", ""]
    pe = bundle.get("pilot_eval")
    if pe:
        L += ["## Pilot answer key, court-held-out (score.py, mentions mapped through redirects)", "",
              "| type | before P | before R | before F1 | after P | after R | after F1 |", "|---|---|---|---|---|---|---|"]
        for t in ("judge", "party", "firm"):
            b, a = pe["scores"][f"before|{t}"], pe["scores"][f"after|{t}"]
            L.append(f"| {t} | {b['precision']:.4f} | {b['recall']:.4f} | {b['f1']:.4f} | {a['precision']:.4f} | {a['recall']:.4f} | {a['f1']:.4f} |")
        L += ["", f"Merges by type in that run: {pe['merges_by_type']}; review queue by type: {pe['review_by_type']}", ""]
    Path(path).write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True, help="V4 output folder (v4_run_ctx.py)")
    ap.add_argument("--output", required=True, help="Layer B output folder (new, or an earlier Layer B output to continue)")
    ap.add_argument("--model", help="saved model.joblib; without it a model is trained on --input")
    ap.add_argument("--llm", choices=["local", "api"], help="review queue pairs with an LLM (off by default)")
    ap.add_argument("--cases", action="append", default=[], help="case JSON folder(s), for LLM docket snippets")
    ap.add_argument("--config", default=str(HERE / "v4_consolidate.yaml"))
    ap.add_argument("--key", help="answer key folder for training/evaluation (default: $V4_ANSWER_KEY, then config reference.answer_key_dir)")
    ap.add_argument("--export", help="also write the final result as exactly two files (entities.jsonl, mentions.jsonl) here")
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config(a.config)
    cfg["_input"] = a.input
    inp, out = Path(a.input).expanduser(), Path(a.output).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    logs = []

    def log(s):
        print(s, flush=True)
        logs.append(s)

    entities = read_jsonl(inp / "entities.jsonl")
    mentions = read_jsonl(inp / "mentions.jsonl")
    uncertain = read_jsonl(inp / "uncertain_pairs.jsonl")
    pending = read_jsonl(inp / "pending_judges.jsonl")
    kd = a.key or os.environ.get("V4_ANSWER_KEY") or cfg["reference"]["answer_key_dir"]
    key_dir = Path(os.path.expanduser(kd)) if kd else None
    if key_dir is not None and not (key_dir / "answer_key_detail.csv").exists():
        log(f"answer key not found in {key_dir}: training without it")
        key_dir = None
    state_p = out / "consolidation_state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {"runs": [], "fingerprints": {}, "pair_scores": {}}
    state["redirects"] = read_jsonl(out / "redirects.jsonl")
    run_no = len(state["runs"]) + 1
    log(f"Layer B run {run_no}: {len(entities)} entities, {len(mentions)} mentions from {inp.name}")

    # model
    if a.model:
        bundle = joblib.load(Path(a.model).expanduser())
        model_path = Path(a.model).expanduser()
    else:
        tw = World(cfg, entities, mentions)
        bundle, _ = train(tw, mentions, cfg, key_dir, log)
        if key_dir is not None:
            ew = World(cfg, entities, mentions)
            ew.set_vectorizer(bundle["vectorizer"])
            bundle["pilot_eval"] = pilot_held_out(ew, mentions, bundle, key_dir, fjc_ids(ew, load_fjc(cfg)), log)
        model_path = out / "model.joblib"
        joblib.dump(bundle, model_path)
        write_model_card(out / "model_card.md", bundle)
        log(f"model saved: {model_path}")
    if bundle.get("version") != MODEL_VERSION:
        sys.exit(f"model version {bundle.get('version')} does not match {MODEL_VERSION}")
    if not (out / "model_card.md").exists():
        write_model_card(out / "model_card.md", bundle)

    log("cards ...")
    world = World(cfg, entities, mentions)
    world.set_vectorizer(bundle["vectorizer"])
    fjc_map = fjc_ids(world, load_fjc(cfg))
    log(f"cards: {len(world.cards)} | {time.time() - t0:.1f}s")
    # incremental: score only pairs touching IDs that are new or got new mentions/cases
    fp = {i: [c["n"], len(c["cases"])] for i, c in world.cards.items()}
    changed = {i for i, v in fp.items() if state["fingerprints"].get(i) != v}
    cands = candidate_pairs(world, bundle["candidate_cut"])
    to_score = sorted((x, y, ct) for (x, y), ct in cands.items()
                      if (x in changed or y in changed or pair_key(x, y) not in state["pair_scores"]))
    log(f"IDs new or changed: {len(changed)}; candidate pairs {len(cands)}; scored now {len(to_score)} | {time.time() - t0:.1f}s")
    Xn = feature_matrix(world, to_score, log)
    pn = bundle["model"].predict_proba(Xn)[:, 1] if len(to_score) else np.zeros(0)
    for (x, y, _), q in zip(to_score, pn):
        state["pair_scores"][pair_key(x, y)] = round(float(q), 6)
    # all current candidate pairs with their (new or stored) scores go to the decision step
    pair_list = sorted((x, y, ct) for (x, y), ct in cands.items())
    probs = np.array([state["pair_scores"][pair_key(x, y)] for x, y, _ in pair_list])
    log(f"scored | {time.time() - t0:.1f}s")
    merges, review, cl = consolidate(world, bundle, pair_list, probs, state, fjc_map, log, run_no)
    log(f"merges decided: {len(merges)} new | {time.time() - t0:.1f}s")

    # LLM on uncertain review pairs (never on refused merges); auto-decides only after its precision passes
    llm_info = None
    llm_notes = {}
    if a.llm:
        llm = LLM(a.llm, cfg, out, log)
        if llm.ok:
            log("LLM: measuring precision on held-out labelled pilot pairs ...")
            llm_info = llm.evaluate(bundle["llm_eval"])
            trusted = llm_info["precision_same"] is not None and llm_info["precision_same"] >= cfg["llm"]["min_precision"]
            llm_info["trusted"] = trusted
            log(f"LLM precision {llm_info['precision_same']} -> {'may decide' if trusted else 'notes only, no auto-merge'}")
            unc = [k for k, why in review if why == "uncertain"][:cfg["llm"]["max_pairs"]]
            ids = sorted({i for k in unc for i in pair_list[k][:2]})
            snips = load_snippets(world, a.cases, ids, cfg)
            for k in unc:
                x, y, ct = pair_list[k]
                ans = llm.ask({"a": card_view(world, x, snips), "b": card_view(world, y, snips),
                               "competing": competing_view(world, x, y, ct)})
                llm_notes[k] = ans
            if trusted:
                keep_review = []
                for k, why in review:
                    ans = llm_notes.get(k)
                    if ans and ans["answer"] == "same":
                        x, y, _ = pair_list[k]
                        ra, rb = cl.root(x), cl.root(y)
                        if ra == rb:
                            continue
                        why2 = cl.refuse(ra, rb, state["pair_scores"], bundle["thresholds"]["T_low"])
                        if why2:
                            keep_review.append((k, f"refused: {why2}"))
                            continue
                        keep, lose = cl.union(ra, rb)
                        merges.append({"old_id": lose, "surviving_id": keep, "score": float(probs[k]), "via_pair": [x, y],
                                       "run": run_no, "decided_by": f"llm:{llm.cfg['local_model'] if a.llm == 'local' else llm.cfg['api_model']}",
                                       "llm_reason": ans["reason"]})
                    elif ans and ans["answer"] == "different":
                        continue
                    else:
                        keep_review.append((k, why if not ans else f"{why}; llm unsure"))
                review = keep_review
            llm_info["calls"] = llm.calls

    # reasons for new merges
    idx = {(pair_list[k][0], pair_list[k][1]): k for k in range(len(pair_list))}
    if merges:
        ks = [idx[tuple(sorted(m["via_pair"]))] for m in merges]
        xs = feature_matrix(world, [pair_list[k] for k in ks])
        for m, rs in zip(merges, explain(bundle, xs, probs[ks])):
            m["reasons"] = rs
    all_redirects = state["redirects"] + merges
    write_jsonl(out / "redirects.jsonl", all_redirects)
    final_of = collapse(all_redirects)

    # outputs
    groups = collections.defaultdict(list)
    for e in entities:
        groups[final_of.get(e["entity_id"], e["entity_id"])].append(e)
    links = related_links(world, pair_list, probs, final_of, bundle)
    write_jsonl(out / "related_links.jsonl", links)
    rel_of = collections.defaultdict(list)
    for r in links:
        rel_of[r["id_a"]].append({"entity_id": r["id_b"], "relation": r["relation"]})
        rel_of[r["id_b"]].append({"entity_id": r["id_a"], "relation": r["relation"]})
    ent_by = {e["entity_id"]: e for e in entities}
    rows = []
    for sid in sorted(groups, key=lambda s: (groups[s][0]["entity_type"], age_key(s))):
        es = sorted(groups[sid], key=lambda e: age_key(e["entity_id"]))
        head = next((e for e in es if e["entity_id"] == sid), es[0])
        rows.append({"entity_id": sid, "entity_type": head["entity_type"],
                     "canonical_name": display_of(world, ent_by, [e["entity_id"] for e in es]),
                     "name_variants": sorted({v for e in es for v in (e.get("name_variants") or [])}),
                     "courts": sorted({v for e in es for v in (e.get("courts") or [])}),
                     "ucids": sorted({v for e in es for v in (e.get("ucids") or [])}),
                     "n_mentions": sum(e.get("n_mentions") or 0 for e in es),
                     "merged_from": [e["entity_id"] for e in es if e["entity_id"] != sid],
                     "related": rel_of.get(sid, []),
                     **({"role_counts": dict(sum((collections.Counter(e.get("role_counts") or {}) for e in es),
                                                  collections.Counter()))} if head["entity_type"] == "party" else {}),
                     **({"unresolved": True} if head.get("unresolved") else {})})
    write_jsonl(out / "entities_consolidated.jsonl", rows)
    write_jsonl(out / "mentions_resolved.jsonl",
                [{"mention_id": m["mention_id"], "entity_type": m.get("entity_type"), "ucid": m.get("ucid"),
                  "raw_name": m.get("raw_name"), "layer_a_id": m.get("entity_id"),
                  "entity_id": final_of.get(m.get("entity_id"), m.get("entity_id"))} for m in mentions])
    rq = []
    for k, why in sorted(review, key=lambda kw: (-probs[kw[0]], pair_list[kw[0]][0], pair_list[kw[0]][1])):
        x, y, ct = pair_list[k]
        r = {"id_a": x, "id_b": y, "entity_type": world.cards[x]["type"], "court": ct, "score": round(float(probs[k]), 6),
             "why": why, "card_a": card_view(world, x), "card_b": card_view(world, y),
             "competing": competing_view(world, x, y, ct)}
        if k in llm_notes:
            r["llm"] = {"answer": llm_notes[k]["answer"], "reason": llm_notes[k]["reason"]}
        rq.append(r)
    write_jsonl(out / "review_queue.jsonl", rq)
    state["fingerprints"] = {i: fp[i] for i in sorted(fp)}
    state["pair_scores"] = {k: state["pair_scores"][k] for k in sorted(state["pair_scores"])}
    state["runs"].append({"run": run_no, "input": inp.name, "entities": len(entities), "mentions": len(mentions),
                          "changed_ids": len(changed), "candidate_pairs": len(cands), "pairs_scored": len(to_score), "new_redirects": len(merges),
                          "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()[:16]})
    redirects_saved = state.pop("redirects")
    state_p.write_text(json.dumps(state, indent=1, sort_keys=True))
    state["redirects"] = redirects_saved
    if a.export:
        export(Path(a.export).expanduser(), rows, mentions, pending, final_of, log)
    elapsed = time.time() - t0
    write_report(out / "report.md", world, entities, rows, all_redirects, merges, rq, bundle, cfg, elapsed, run_no,
                 pending, uncertain, final_of, llm_info, len(changed), len(to_score), len(cands), links)
    log(f"done in {elapsed:.1f}s: {len(merges)} new merges, {len(all_redirects)} redirects in total, review queue {len(rq)}")


def export(folder, rows, mentions, pending, final_of, log):
    org_votes = collections.defaultdict(lambda: [0, 0])     # final entity -> [person mentions, organisation mentions]
    for m in mentions:
        ctx = m.get("context") or {}
        fid = final_of.get(m.get("entity_id"), m.get("entity_id"))
        org_votes[fid][1 if (ctx.get("is_organisation") or ctx.get("is_org")) else 0] += 1
    """Exactly two files: entities.jsonl (final entities after Layer B) and mentions.jsonl (every mention with its
    details, role / side and FINAL entity ID; pending docket judge mentions have entity_id null)."""
    folder.mkdir(parents=True, exist_ok=True)
    ents = []
    for r in rows:
        e = {"entity_id": r["entity_id"], "entity_type": r["entity_type"], "canonical_name": r["canonical_name"],
             "name_variants": r["name_variants"], "courts": r["courts"], "cases": r["ucids"], "n_mentions": r["n_mentions"],
             "merged_from": r["merged_from"], "related": r["related"]}
        if r["entity_type"] == "party":
            rc = r.get("role_counts") or {}
            e["plaintiff_side"] = rc.get("plaintiff-side", 0)
            e["defendant_side"] = rc.get("defendant-side", 0)
            e["other_side"] = sum(v for k, v in rc.items() if k not in ("plaintiff-side", "defendant-side"))
        if r["entity_type"] in ("party", "counsel"):
            pv, ov = org_votes[r["entity_id"]]
            e["is_organisation"] = ov > pv     # counsel: an office named as counsel; party: the name gate's decision
        if r.get("unresolved"):
            e["unresolved"] = True
        ents.append(e)
    write_jsonl(folder / "entities.jsonl", ents)
    out = []
    for m in mentions:
        ctx = m.get("context") or {}
        out.append({"mention_id": m["mention_id"], "entity_type": m.get("entity_type"), "raw_name": m.get("raw_name"),
                    "clean_name": m.get("clean_name"), "case": m.get("ucid"), "court": m.get("court"),
                    "role": m.get("role", []), "side": m.get("side", []),
                    "is_organisation": bool(ctx.get("is_organisation") or ctx.get("is_org")) if m.get("entity_type") in ("party", "counsel") else m.get("entity_type") == "firm",
                    "context": ctx,
                    "layer_a_entity_id": m.get("entity_id"),
                    "entity_id": final_of.get(m.get("entity_id"), m.get("entity_id"))})
    for m in pending:
        out.append({"mention_id": None, "entity_type": "judge", "raw_name": m.get("raw_name"), "clean_name": m.get("clean_name"),
                    "case": m.get("ucid"), "court": m.get("court"), "role": [], "side": [], "is_organisation": False,
                    "context": m.get("context") or {},
                    "layer_a_entity_id": None, "entity_id": None, "status": "pending docket judge (no judge ID yet)"})
    write_jsonl(folder / "mentions.jsonl", out)
    for extra in sorted(folder.iterdir()):
        if extra.name not in ("entities.jsonl", "mentions.jsonl"):
            log(f"export: note - {extra.name} in {folder} is not written by --export")
    log(f"export: {len(ents)} entities, {len(out)} mentions -> {folder}")


def write_report(path, world, entities, rows, all_redirects, new_merges, rq, bundle, cfg, elapsed, run_no, pending,
                 uncertain, final_of, llm_info, n_changed, n_scored, n_cands, links=()):
    ent = {e["entity_id"]: e for e in entities}
    before = collections.Counter(e["entity_type"] for e in entities if not e.get("unresolved"))
    after = collections.Counter(r["entity_type"] for r in rows if not r.get("unresolved"))
    th = bundle["thresholds"]
    L = ["# Layer B consolidation report", "",
         f"- Run {run_no}; model `{bundle['model_name']}`; T_high by kind "
         + ", ".join(f"{k} {('%.4f' % v) if v is not None else 'none (review only)'}" for k, v in th["T_high_by_kind"].items())
         + f"; T_low {th['T_low']:.4f}",
         f"- IDs new or changed since the last run: {n_changed}; candidate pairs {n_cands}; pairs scored in this run {n_scored}",
         f"- Time: {elapsed:.1f} s", f"- Pending docket judge mentions in the input (not IDs, left as they are): {len(pending)}",
         f"- Layer A uncertain pairs in the input: {len(uncertain)}", "",
         "## IDs by type", "", "| type | before (Layer A) | after (Layer B) | merged away |", "|---|---|---|---|"]
    for t in TYPES:
        L.append(f"| {t} | {before[t]} | {after[t]} | {before[t] - after[t]} |")
    rq_t = collections.Counter(r["entity_type"] for r in rq)
    rq_w = collections.Counter((r["entity_type"], r["why"].split(":")[0]) for r in rq)
    rel = collections.Counter((r["entity_type"], r["relation"]) for r in links)
    L += ["", f"## Related links ({len(links)}; both IDs kept, in related_links.jsonl)", "",
          "| type | office | parent_or_subsidiary |", "|---|---|---|"]
    for t in ("party", "firm"):
        L.append(f"| {t} | {rel[(t, 'office')]} | {rel[(t, 'parent_or_subsidiary')]} |")
    # organisation names identical after normalisation but still split (same core + office, different final IDs)
    same = collections.defaultdict(set)
    for r in rows:
        if r["entity_type"] == "firm" or (r["entity_type"] == "party" and r["entity_id"] in world.cards
                                           and not world.cards[r["entity_id"]]["is_person"]):
            c = world.cards.get(r["entity_id"])
            if c:
                same[(r["entity_type"], c["org_core"], c["org_office"])].add(r["entity_id"])
    split = sorted(((k, v) for k, v in same.items() if len(v) > 1), key=lambda kv: (-len(kv[1]), kv[0]))
    L += ["", f"## Organisation names identical after normalisation but still split: {len(split)} names, "
              f"{sum(len(v) for _, v in split)} IDs", "",
          "Same normalised core name and office, different final IDs (Layer A needs shared contact details for firms and "
          "never joins two parties of one case; Layer B merges organisations only in evidence groups that reached the "
          "precision target).", "", "| type | normalised name | office | IDs |", "|---|---|---|---|"]
    for (t, core, office), v in split[:30]:
        L.append(f"| {t} | {core} | {office} | {len(v)} |")
    L += ["", "## Review queue by type", "", "| type | pairs | uncertain | refused by a safety check |", "|---|---|---|---|"]
    for t in TYPES:
        L.append(f"| {t} | {rq_t[t]} | {rq_w[(t, 'uncertain')]} | {rq_w[(t, 'refused')]} |")

    def nm(i):
        e = ent.get(i)
        return e.get("canonical_name") if e else i

    def line(r):
        rs = "; ".join(f"{x['feature']}={x['value']}" for x in r.get("reasons", []))
        return (f"| {r['old_id']} | {nm(r['old_id'])} | {final_of.get(r['old_id'], r['surviving_id'])} | "
                f"{nm(final_of.get(r['old_id'], r['surviving_id']))} | {r['score']:.4f} | {r.get('decided_by', '')} | {rs} | {r['run']} |")
    hdr = ["| old ID | old name | surviving ID | surviving name | score | decided by | top reasons | run |", "|---|---|---|---|---|---|---|---|"]
    judge_m = [r for r in all_redirects if ent.get(r["old_id"], {}).get("entity_type") == "judge"]
    L += ["", f"## Every judge merge ({len(judge_m)})", ""] + hdr + [line(r) for r in sorted(judge_m, key=lambda r: (nm(r["old_id"]) or "", r["old_id"]))]
    rng = random.Random(cfg["report"]["seed"])
    for t in ("party", "counsel", "firm"):
        ms = sorted((r for r in all_redirects if ent.get(r["old_id"], {}).get("entity_type") == t), key=lambda r: r["old_id"])
        pick = sorted(rng.sample(ms, min(cfg["report"]["sample_per_type"], len(ms))), key=lambda r: r["old_id"])
        L += ["", f"## {t}: {len(pick)} random merges of {len(ms)}", ""] + hdr + [line(r) for r in pick]
    if llm_info:
        L += ["", "## LLM", "", "```", json.dumps(llm_info, indent=1, sort_keys=True), "```"]
    pe = bundle.get("pilot_eval")
    if pe:
        L += ["", "## Pilot answer key, court-held-out (score.py resolution, mentions mapped through redirects)", "",
              "Each court is scored by a fold model that never saw that court.", "",
              "| type | before P | before R | before F1 | after P | after R | after F1 |", "|---|---|---|---|---|---|---|"]
        for t in ("judge", "party", "firm"):
            b, a2 = pe["scores"][f"before|{t}"], pe["scores"][f"after|{t}"]
            L.append(f"| {t} | {b['precision']:.4f} | {b['recall']:.4f} | {b['f1']:.4f} | {a2['precision']:.4f} | {a2['recall']:.4f} | {a2['f1']:.4f} |")
    L += ["", "Model details (features, weights, training pairs, thresholds): model_card.md", ""]
    Path(path).write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

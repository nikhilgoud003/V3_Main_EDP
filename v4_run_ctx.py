#!/usr/bin/env python3
"""V4 incremental entity resolution with CONTEXT-AWARE matching (local).

A copy of v4_run.py. Unchanged: extraction (which names become mentions), normalization,
ID minting, output files, and the incremental behaviour (processed files skipped by
filename + sha256; old IDs never change; saved entities are never merged together).

Changed: matching. Each new mention is compared only with candidates from an index keyed by
(entity_type, surname / exact name), and merges need context:
  - never across entity types;
  - judges: same surname + compatible first name (initial or nickname) + same court; if more
    than one saved judge in that court fits, keep apart and log;
  - parties: identical normalized name in another case AND evidence (shared counsel or firm
    entity, shared co-party, or a rare name); never two different names in one case; Doe
    names never across cases;
  - counsel: compatible name + same email, phone, office address or firm name;
  - firms: identical normalized name (legal forms already removed) + same address, phone or
    email domain; U.S. Attorney / Federal Public Defender offices kept per district (whole-word
    match);
  - close pairs that are not proven are written to uncertain_pairs.jsonl and not merged.

Outputs in --output: entities.jsonl, mentions.jsonl (each mention also carries its context),
state.json, uncertain_pairs.jsonl.

Usage:
  python v4_run_ctx.py --input PATH [PATH ...] --output DIR
"""

import argparse
import faulthandler
import hashlib
import json
import os
import re
import signal
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

# --- RULES: every word list, pattern and threshold comes from v4_rules.yaml (see the `source:` notes there) ---
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4_names as N  # noqa: E402  (shared name gate / organisation normalisation / display names)
import v4_readers as READERS  # noqa: E402  (document + generic JSON readers)

RULES_FILE = Path(__file__).resolve().parent / "v4_rules.yaml"
RULES = {}


def _word_re(words):
    return r"\b(" + "|".join(words) + r")\b"


def load_fjc_names(path):
    """(first-or-middle, last) pairs and (initial, last) pairs from the FJC judge list."""
    import csv
    names = set()
    p = Path(path)
    if not p.exists():
        print(f"WARNING: FJC list not found at {p}; docket judges then need case fields or the case count", flush=True)
        return names
    norm = lambda x: " ".join(re.sub(r"[^a-z ]", " ", (x or "").lower()).split())
    with open(p, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            last = norm(row.get("Last Name")).split()
            if not last:
                continue
            last = last[-1]
            for given in (norm(row.get("First Name")), norm(row.get("Middle Name"))):
                for g in given.split()[:1]:
                    names.add((g, last))
                    names.add((g[0], last))
    return names


def load_first_name_like(male, female, surnames):
    """Census 1990: words more frequent as a first name than as a surname."""
    def read(p):
        out = {}
        p = Path(p)
        if not p.exists():
            print(f"WARNING: name list not found: {p}", flush=True)
            return out
        for line in open(p, encoding="utf-8", errors="replace"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    out[parts[0].lower()] = float(parts[1])
                except ValueError:
                    pass
        return out
    m, f, sn = read(male), read(female), read(surnames)
    return {n for n in set(m) | set(f) if max(m.get(n, 0), f.get(n, 0)) > sn.get(n, 0)}


def load_rules(path=None):
    """Load v4_rules.yaml and build the module-level lists, sets and compiled patterns."""
    global RULES, RULES_FILE, EXTRACTION_CONFIG, DOCKET_JUDGE_PATTERNS, GATE_CFG, DOC_CFG, RAWINFO_STAFF_RE
    global RARE_MAX_CASES, RARE_PERSON_MIN_TOKENS, COPARTY_MAX_CASES, COPARTY_MAX_PARTIES_IN_CASE, PROGRESS_EVERY
    global WEBMAIL_DOMAINS, NAME_SUFFIXES, COUNSEL_TAG_TOKENS, GOV_PARTY_NAMES, GOV_NOT_IF_CONTAINS
    global PLACEHOLDER_RE, USAO_RE, FPD_RE, NICKNAMES, DOE_PATTERN, TITLE_PATTERN
    global JUDGE_TITLE_PATTERN, APOSTROPHE_RE, PHONE_MIN_DIGITS, PHONE_COUNTRY_CODE, ZIP_RE, STREET_NUMBER_RE
    global UNCERTAIN_MAX_CANDIDATES, UNCERTAIN_MAX_COURTS, MATCHING_LABEL
    global JUDGE_PARENS_RE, JUDGE_TITLE_EVIDENCE, TYPO_MAX_EDITS, TYPO_MIN_LEN, TYPO_BOTH_FREQ
    global SURNAME_ONLY_LEVELS, UNRESOLVED_PREFIX
    global COUNSEL_FULL_MIN_TOKENS, FJC_NAMES, FIRST_NAME_LIKE
    global PLACEHOLDERS, FIRM_PLACEHOLDER_PREFIXES, BAR_RE
    global SIDE_BY_PTYPE, SIDE_BY_ROLE, DEFAULT_SIDE, JUNK_MUST_CONTAIN
    global DOCKET_CREATE_MIN_CASES, DOCKET_TITLE_RE, DOCKET_TRAILING_WORDS
    global ACR_MIN, ACR_MAX, INITIALS_SKIP
    if path:
        RULES_FILE = Path(path).expanduser()
    with open(RULES_FILE, encoding="utf-8") as f:
        R = yaml.safe_load(f)
    RULES = R
    EXTRACTION_CONFIG = {t: dict(R["entity_types"][t]) for t in ("judge", "party", "counsel", "firm")}
    DOCKET_JUDGE_PATTERNS = [re.compile(p) for p in R["docket_judge_patterns"]["patterns"]]
    pe = R["party_evidence"]
    RARE_MAX_CASES = pe["rare_max_cases"]
    RARE_PERSON_MIN_TOKENS = pe["rare_person_min_tokens"]
    COPARTY_MAX_CASES = pe["coparty_max_cases"]
    COPARTY_MAX_PARTIES_IN_CASE = pe["coparty_max_parties_in_case"]
    PROGRESS_EVERY = R["progress_every"]["files"]
    WEBMAIL_DOMAINS = set(R["webmail_domains"]["domains"])
    NAME_SUFFIXES = set(R["name_suffixes"]["words"])
    COUNSEL_TAG_TOKENS = set(R["counsel_tag_tokens"]["words"])
    GOV_PARTY_NAMES = set(R["government_party_names"]["names"])
    GOV_NOT_IF_CONTAINS = R["government_party_names"]["not_if_contains"]
    ph = R["placeholder_party"]
    PLACEHOLDER_RE = re.compile(_word_re(ph["words"]) + "|^(" + "|".join(ph["exact_names"]) + ")$")
    USAO_RE = re.compile(R["federal_offices"]["usao_pattern"])
    FPD_RE = re.compile(R["federal_offices"]["fpd_pattern"])
    NICKNAMES = defaultdict(set)
    for g in R["nickname_groups"]["groups"]:
        names = g.split()
        for n in names:
            NICKNAMES[n].update(names)
    DOE_PATTERN = re.compile(R["doe_pattern"]["pattern"])
    TITLE_PATTERN = re.compile(_word_re(R["title_words"]["words"]))
    J = R["judges"]
    JUDGE_TITLE_PATTERN = re.compile(_word_re(R["judge_title_words"]["words"] + J["title_typos"]["words"]))
    JUDGE_PARENS_RE = re.compile(J["strip_parenthetical"]["pattern"])
    JUDGE_TITLE_EVIDENCE = {t: list(ws) for t, ws in J["title_evidence"]["titles"].items()}
    TYPO_MAX_EDITS = J["typo"]["max_first_name_edits"]
    TYPO_MIN_LEN = J["typo"]["min_first_name_length"]
    TYPO_BOTH_FREQ = J["typo"]["both_frequent_min_mentions"]
    SURNAME_ONLY_LEVELS = list(J["surname_only"]["levels"])
    UNRESOLVED_PREFIX = J["surname_only"]["unresolved_id_prefix"]
    O = R["organisations"]
    ACR_MIN, ACR_MAX = O["acronym"]["min_length"], O["acronym"]["max_length"]
    INITIALS_SKIP = set(O["acronym"]["initials_skip_words"])
    COUNSEL_FULL_MIN_TOKENS = R["counsel"]["no_contact"]["full_name_min_tokens"]
    rel = lambda x: (Path(x).expanduser() if Path(x).expanduser().is_absolute() else RULES_FILE.parent / x)
    FJC_NAMES = load_fjc_names(rel(R["docket_judges"]["fjc_list"]["path"]))
    fn = R["docket_judges"]["first_name_only"]
    FIRST_NAME_LIKE = load_first_name_like(rel(fn["male_first"]), rel(fn["female_first"]), rel(fn["surnames"]))
    G = R["type_gate"]
    RAWINFO_STAFF_RE = re.compile(_word_re([re.escape(w) for w in G["rawinfo_staff_phrases"]["phrases"]]))
    PLACEHOLDERS = {t: set(G["placeholders"][t]) for t in ("counsel", "firm", "party")}
    FIRM_PLACEHOLDER_PREFIXES = tuple(G["placeholders"]["firm_prefixes"])
    BAR_RE = re.compile(G["bar_pattern"]["pattern"], re.I)
    JUNK_MUST_CONTAIN = re.compile(G["junk_rule"]["must_contain_pattern"])
    DJ = R["docket_judges"]
    DOCKET_CREATE_MIN_CASES = DJ["rule"]["min_cases"]
    DOCKET_TITLE_RE = re.compile(_word_re([re.escape(w) for w in DJ["rule"]["title_words"]]), re.I)
    DOCKET_TRAILING_WORDS = set(DJ["trailing_words"]["words"])
    GATE_CFG = R["name_gate"]
    DOC_CFG = R["documents"]
    N.TRAILING_ORG_TYPE_IS_LEGAL = O["legal_form_rule"]["value"] == "legal_type_and_trailing_org_type"
    N.org_parts.cache_clear()
    PR = R["party_roles"]
    SIDE_BY_PTYPE, SIDE_BY_ROLE, DEFAULT_SIDE = dict(PR["side_by_party_type"]), dict(PR["side_by_role"]), PR["default_side"]
    APOSTROPHE_RE = re.compile(R["apostrophes"]["pattern"])
    c = R["contact"]
    PHONE_MIN_DIGITS = c["phone_min_digits"]
    PHONE_COUNTRY_CODE = str(c["phone_country_code"])
    ZIP_RE = re.compile(c["zip_pattern"])
    STREET_NUMBER_RE = re.compile(c["street_number_pattern"])
    UNCERTAIN_MAX_CANDIDATES = R["uncertain_log"]["max_candidates_per_mention"]
    UNCERTAIN_MAX_COURTS = R["uncertain_log"]["max_courts_shown"]
    MATCHING_LABEL = R["matching_label"]
    return R


load_rules()

# --- 2. PERSISTENT STORAGE & STATE MANAGEMENT ---
STORAGE_DIR = None
ENTITIES_FILE = None
MENTIONS_FILE = None
STATE_FILE = None
UNCERTAIN_FILE = None
DROPPED_FILE = None
PENDING_FILE = None
GATE_CACHE_FILE = None
GATE_REVIEW_FILE = None
DOC_CACHE_FILE = None
DOC_REVIEW_FILE = None
QUIET_UNCERTAIN = False   # set while re-checking pending judges, so uncertain pairs are not logged twice
GATE = None        # v4_names.NameGate, created in main()
EXTRACTOR = None   # v4_readers.Extractor, created on the first document / generic file
DOC_MODEL = "gliner2"
LLM_CFG = None     # local Ollama settings when --llm local is given
FIELD_OF = {"party": "parties[].name", "counsel": "parties[].counsel[].name",
            "firm": "parties[].counsel[].entity_info.office_name", "judge": "judge / referred_judges / parties[].judge"}


class MemoryManager:
    def __init__(self, storage_dir=None):
        self.storage_dir = storage_dir
        self.state = {
            "counters": {"judge": 1, "party": 1, "counsel": 1, "firm": 1},
            "processed_files": {}  # filename: sha256
        }
        self.entities = {}   # entity_id: entity_dict
        self.mentions = {}   # mention_id: mention_dict
        self.uncertain = []  # uncertain pair rows
        self.dropped = []    # dropped item rows (type gate, placeholders, docket rules, ...)
        self.pending = []     # docket / one-word judge mentions waiting to join or create a judge
        self.gate_review = []  # names the gate could not decide (no agreement, no known entity, no LLM)
        self.doc_review = []   # document / generic reader rows to check (low confidence, case number mismatch)
        self.load()

    def load(self):
        if STATE_FILE.exists():
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                self.state = json.load(f)
        if ENTITIES_FILE.exists():
            with open(ENTITIES_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        ent = json.loads(line)
                        self.entities[ent["entity_id"]] = ent
        if MENTIONS_FILE.exists():
            with open(MENTIONS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        mnt = json.loads(line)
                        self.mentions[mnt["mention_id"]] = mnt
        if UNCERTAIN_FILE.exists():
            with open(UNCERTAIN_FILE, "r", encoding="utf-8") as f:
                self.uncertain = [json.loads(l) for l in f if l.strip()]
        if DROPPED_FILE.exists():
            with open(DROPPED_FILE, "r", encoding="utf-8") as f:
                self.dropped = [json.loads(l) for l in f if l.strip()]
        if PENDING_FILE.exists():
            with open(PENDING_FILE, "r", encoding="utf-8") as f:
                self.pending = [json.loads(l) for l in f if l.strip()]
        for attr, fp in (("gate_review", GATE_REVIEW_FILE), ("doc_review", DOC_REVIEW_FILE)):
            if fp.exists():
                with open(fp, "r", encoding="utf-8") as f:
                    setattr(self, attr, [json.loads(l) for l in f if l.strip()])
        if self.mentions and not all("context" in m for m in self.mentions.values()):
            raise SystemExit("ERROR: this output folder was made by v4_run.py (mentions have no context). "
                             "Use a new --output folder for v4_run_ctx.py.")
        print(f"Memory Loaded: {len(self.entities)} entities, {len(self.mentions)} mentions, "
              f"{len(self.state['processed_files'])} processed files.", flush=True)

    def save(self):
        """Every file is written to a temp file and renamed over the old one, so an interrupted save never
        leaves a half-written output; state.json goes last (it marks the files as processed)."""
        self.state["matching"] = MATCHING_LABEL
        self.state.pop("docket_judge_cases", None)
        t0 = time.perf_counter()
        for label, fp, rows in (("entities", ENTITIES_FILE, self.entities.values()),
                                ("mentions", MENTIONS_FILE, self.mentions.values()),
                                ("uncertain pairs", UNCERTAIN_FILE, self.uncertain),
                                ("dropped entities", DROPPED_FILE, self.dropped),
                                ("pending judges", PENDING_FILE, self.pending),
                                ("gate review", GATE_REVIEW_FILE, self.gate_review),
                                ("document review", DOC_REVIEW_FILE, self.doc_review)):
            N.atomic_write_lines(fp, rows)
            print(f"End step: saved {label} ({fp.name}) | {time.perf_counter() - t0:.1f}s", flush=True)
        if GATE is not None:
            GATE.save()
        if EXTRACTOR is not None:
            EXTRACTOR.save()
        N.atomic_write_text(STATE_FILE, json.dumps(self.state, indent=2))
        print(f"End step: saved state.json | {time.perf_counter() - t0:.1f}s", flush=True)

    def get_next_id(self, entity_type):
        cfg = EXTRACTION_CONFIG[entity_type]
        idx = self.state["counters"][entity_type]
        self.state["counters"][entity_type] += 1

        prefix = cfg["id_prefix"]
        entity_id = f"ent_{prefix}_{idx:06d}"
        specific_id = f"{prefix}{idx:06d}"
        return entity_id, specific_id


memory = None  # created in main() once the output folder is known

# --- 3. NORMALIZATION (patterns from v4_rules.yaml) ---
def clean_text_base(text):
    if not text:
        return ""
    text = str(text).lower()
    text = APOSTROPHE_RE.sub("", text)  # O'Brien -> obrien
    text = TITLE_PATTERN.sub("", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split()).strip()


def normalize_entity_name(raw_name, entity_type):
    if entity_type == "judge" and raw_name:
        raw_name = JUDGE_PARENS_RE.sub(" ", str(raw_name))
    base = clean_text_base(raw_name)
    if base and entity_type == "judge":
        base = " ".join(JUDGE_TITLE_PATTERN.sub(" ", base).split())
    if not base:
        return ""
    if entity_type in ["firm", "party"]:
        # one organisation normalisation for both layers (v4_names.org_parts): &/and/+, punctuation and line breaks
        # treated alike, legal forms removed with the parser's own components, office text moved out
        return N.org_parts(raw_name)["core"] or base
    return base


def is_doe_entity(clean_name):
    return bool(DOE_PATTERN.match(clean_name))


# --- 4. CONTEXT HELPERS ---
def norm_email(e):
    e = (e or "").strip().lower()
    return e if "@" in e else ""


def email_domain(e):
    e = norm_email(e)
    d = e.split("@", 1)[1] if e else ""
    return "" if d in WEBMAIL_DOMAINS else d


def norm_phone(p):
    d = re.sub(r"\D", "", p or "")
    if len(d) == 10 + len(PHONE_COUNTRY_CODE) and d.startswith(PHONE_COUNTRY_CODE):
        d = d[len(PHONE_COUNTRY_CODE):]
    return d if len(d) >= PHONE_MIN_DIGITS else ""


def norm_address(a):
    """Street number + ZIP when both exist (robust to formatting); else the cleaned text."""
    a = (a or "").lower()
    if not a.strip():
        return ""
    zips = ZIP_RE.findall(a)
    num = STREET_NUMBER_RE.search(a)
    if zips and num and num.group(1) != zips[-1]:
        return f"{num.group(1)}|{zips[-1]}"
    return " ".join(re.sub(r"[^\w\s]", " ", a).split())


def federal_agency(clean_name):
    """Whole-word match on the cleaned firm name."""
    if USAO_RE.search(clean_name):
        return "USAO"
    if FPD_RE.search(clean_name):
        return "FPD"
    return None


def person_tokens(clean_name):
    toks = clean_name.split()
    sfx = {t for t in toks if t in NAME_SUFFIXES}
    core = [t for t in toks if t not in NAME_SUFFIXES]
    return core, sfx


def surname(clean_name):
    core, _ = person_tokens(clean_name)
    return core[-1] if core else ""


def first_names_compatible(f1, f2):
    if f1 == f2:
        return True
    if len(f1) == 1 and f2.startswith(f1):
        return True
    if len(f2) == 1 and f1.startswith(f2):
        return True
    return f2 in NICKNAMES.get(f1, ())


def counsel_core(clean_name):
    """Counsel name without trailing office tags (keeps at least two words)."""
    toks = clean_name.split()
    while len(toks) > 2 and toks[-1] in COUNSEL_TAG_TOKENS:
        toks.pop()
    return " ".join(toks)


def name_core(clean_name, ent_type):
    return counsel_core(clean_name) if ent_type == "counsel" else clean_name


def entity_fits(mname, ent, ent_type):
    """Item 3 (chaining guard): compatible with at least one name of the entity AND with every
    full first name already in it, and no conflicting generational suffix."""
    names = [name_core(v, ent_type) for v in ent["name_variants"]]
    if not any(person_names_compatible(mname, v) for v in names):
        return False
    cm, sm = person_tokens(mname)
    if len(cm) < 2:
        return True
    for v in names:
        cv, sv = person_tokens(v)
        if len(cv) >= 2 and len(cv[0]) > 1 and not first_names_compatible(cm[0], cv[0]):
            return False
        if sm and sv and sm != sv:
            return False
    return True


def is_surname_only(clean_name):
    core, _ = person_tokens(clean_name)
    return len(core) == 1


# --- Step 2 judge helpers ---
from rapidfuzz.distance import OSA

UNRESOLVED = "__UNRESOLVED__"  # sentinel returned by match_judge


def judge_display_name(raw_name):
    """Judge name for display: no brackets, no titles ("Magistrate Judge X (Settlement)" -> "X")."""
    raw = JUDGE_PARENS_RE.sub(" ", str(raw_name))
    titles = set(RULES["title_words"]["words"]) | set(RULES["judge_title_words"]["words"]) | set(RULES["judges"]["title_typos"]["words"])
    titles = {t.replace(" ", "") for t in titles}
    keep = [t for t in raw.split() if re.sub(r"[^a-z]", "", t.lower()) not in titles]
    return " ".join(keep).strip(" ,;.-") or str(raw_name).strip()


def side_of(party_type, role):
    if party_type in SIDE_BY_PTYPE:
        return SIDE_BY_PTYPE[party_type]
    return SIDE_BY_ROLE.get(role, DEFAULT_SIDE)


def judge_titles(raw_name):
    s = " " + " ".join(re.sub(r"[^a-z0-9]+", " ", str(raw_name).lower()).split()) + " "
    return sorted(t for t, words in JUDGE_TITLE_EVIDENCE.items() if any(f" {w} " in s for w in words))


def case_division(ucid):
    cid = ucid.split(";;", 1)[-1]
    return cid.split(":", 1)[0] if ":" in cid else ""


def entity_divisions(ent):
    return {(u.split(";;", 1)[0], case_division(u)) for u in ent["ucids"]}


def title_pick(m, cands):
    """Title as evidence: among several candidates, the only one sharing the mention's title."""
    mt = set(m["context"].get("titles", []))
    if not mt:
        return None
    hit = [e for e in cands if mt & INDEX.ctx[e["entity_id"]]["titles"]]
    return hit[0] if len(hit) == 1 else None


def pick_by_levels(m, pool):
    """Same case -> same division -> same court. Returns (entity, level, ambiguous_candidates)."""
    div = (m["court"], case_division(m["ucid"]))
    tests = {"case": lambda e: m["ucid"] in e["ucids"],
             "division": lambda e: div in entity_divisions(e),
             "court": lambda e: m["court"] in e["courts"]}
    for level in SURNAME_ONLY_LEVELS:
        c = [e for e in pool if tests[level](e)]
        if len(c) == 1:
            return c[0], level, []
        if len(c) > 1:
            t = title_pick(m, c)
            return (t, level, []) if t else (None, level, c)
    return None, None, []


def typo_fits(name, ent, court):
    """Same surname, first name one edit from every full first name in the entity (at least one real typo),
    not both spellings frequent, no conflicting suffix."""
    cm, sm = person_tokens(name)
    if len(cm) < 2 or len(cm[0]) < TYPO_MIN_LEN:
        return False
    sur = cm[-1]
    fulls, typo_seen = [], False
    for v in ent["name_variants"]:
        cv, sv = person_tokens(v)
        if sm and sv and sm != sv:
            return False
        if len(cv) < 2 or len(cv[0]) <= 1:
            continue
        if cv[-1] != sur:
            return False
        fulls.append(cv[0])
    if not fulls:
        return False
    for f in fulls:
        if first_names_compatible(cm[0], f):
            continue
        if len(f) < TYPO_MIN_LEN or OSA.distance(cm[0], f) > TYPO_MAX_EDITS:
            return False
        if (INDEX.judge_first_counts[(court, sur, cm[0])] >= TYPO_BOTH_FREQ
                and INDEX.judge_first_counts[(court, sur, f)] >= TYPO_BOTH_FREQ):
            return False
        typo_seen = True
    return typo_seen


def person_names_compatible(a, b):
    """Same surname, compatible first name, no conflicting generational suffix."""
    ca, sa = person_tokens(a)
    cb, sb = person_tokens(b)
    if not ca or not cb:
        return False
    if len(ca) < 2 or len(cb) < 2:
        return a == b
    if ca[-1] != cb[-1]:
        return False
    if sa and sb and sa != sb:
        return False
    return first_names_compatible(ca[0], cb[0])


def is_gov_party(clean):
    return clean in GOV_PARTY_NAMES and GOV_NOT_IF_CONTAINS not in clean


def is_placeholder_party(clean):
    return bool(is_doe_entity(clean) or PLACEHOLDER_RE.search(clean))


# --- Step 4: organisation keys and acronyms (normalisation and the office field come from v4_names) ---
def is_acronym(raw_part):
    t = str(raw_part).strip().replace(".", "")
    return t.isalpha() and t.isupper() and ACR_MIN <= len(t) <= ACR_MAX and " " not in t


def org_key(raw_name):
    """Matching key for an organisation: normalised core name plus its office ("<core> | <office>")."""
    return N.org_key(raw_name)


def org_initials(key):
    toks = [t for t in key.split("|")[0].split() if t not in INITIALS_SKIP]
    ini = "".join(t[0] for t in toks)
    return ini if len(toks) >= 2 and ACR_MIN <= len(ini) <= ACR_MAX else ""


def org_batch_initials(raw_name, clean, kind):
    """Initials an organisation party adds to its court's acronym set: its core name and a multi-word office."""
    if kind != N.ORG or is_gov_party(clean) or is_placeholder_party(clean) or is_acronym(N.org_head(raw_name)):
        return []
    parts = N.org_parts(raw_name)
    return [x for x in (org_initials(parts["core"]), org_initials(parts["office"])) if x]


def party_org_decision(raw_name, clean, court, kind):
    """Context fields for a party at match time: is_org, and org_key or acronym. kind = the name gate's decision."""
    if is_gov_party(clean) or is_placeholder_party(clean):
        return {"is_org": []}
    parts = N.org_parts(raw_name)
    office = {"office": [parts["office"]]} if parts["office"] else {}
    head = N.org_head(raw_name)
    if is_acronym(head):
        acr = head.replace(".", "").lower()
        if parts["office"] or acr in INDEX.org_initials.get(court, ()):
            return {"is_org": [True], "acronym": [acr], **office}
    if kind == N.ORG:
        return {"is_org": [True], "org_key": [org_key(raw_name)], **office}
    return {"is_org": []}


# --- 5. CONTEXT-AWARE INDEX ---
class ContextIndex:
    """Candidate index and per-entity evidence, rebuilt from saved mentions on load."""

    def __init__(self):
        self.by_key = defaultdict(list)          # (kind, key...) -> [entity_id] in creation order
        self.ctx = defaultdict(lambda: defaultdict(set))  # entity_id -> field -> set
        self.party_name_cases = defaultdict(set)  # normalized party name -> ucids (saved + batch)
        self.judge_first_counts = defaultdict(int)  # (court, surname, first name) -> judge mentions (saved + batch)
        self.org_initials = defaultdict(set)  # court -> initials of organisation party names (saved + batch)
        self.counsel_name_cases = defaultdict(set)  # counsel name (tags stripped) -> cases (saved + batch)
        self.judges_by_case = defaultdict(set)  # ucid -> judge entity IDs (not unresolved records)

    def keys_for(self, ent_type, clean, court, unresolved=False, ctx=None):
        ctx = ctx or {}
        if ent_type == "party" and not is_gov_party(clean):
            if ctx.get("acronym"):
                return [("party_acronym", court, ctx["acronym"][0])]
            if ctx.get("org_key"):
                k = ctx["org_key"][0]
                ini = org_initials(k)
                ini = ini if " | " not in k else ""   # an office of an organisation is not the organisation's acronym
                return [("party", k)] + ([("party_initials", court, ini)] if ini else [])
        if ent_type == "judge" and unresolved:
            return [("judge_unresolved", court, surname(clean))]
        if ent_type in ("judge", "counsel"):
            return [(ent_type, surname(name_core(clean, ent_type)))]
        if ent_type == "party":
            return [("party_gov",)] if is_gov_party(clean) else [("party", clean)]
        fed = federal_agency(clean)
        if fed:
            return [("firm_fed", fed, court)]
        return [("firm", clean, (ctx.get("office") or [""])[0])]

    def register(self, ent_id, mention):
        """Add a mention's name keys and context to its entity."""
        t, clean, court = mention["entity_type"], mention["clean_name"], mention["court"]
        for k in self.keys_for(t, clean, court, mention.get("unresolved", False), mention.get("context")):
            lst = self.by_key[k]
            if ent_id not in lst:
                lst.append(ent_id)
        if t == "judge" and not mention.get("unresolved") and not str(ent_id).startswith(f"ent_{UNRESOLVED_PREFIX}_"):
            self.judges_by_case[mention["ucid"]].add(ent_id)
        c = self.ctx[ent_id]
        for field, vals in (mention.get("context") or {}).items():
            if isinstance(vals, list):
                c[field].update(v for v in vals if v)

    def add_party_name_case(self, clean, ucid):
        self.party_name_cases[clean].add(ucid)

    def add_org_initials(self, ctx, court):
        for ini in (ctx or {}).get("batch_initials", []):
            if ini:
                self.org_initials[court].add(ini)

    def add_judge_first(self, clean, court):
        core, _ = person_tokens(clean)
        if len(core) >= 2:
            self.judge_first_counts[(court, core[-1], core[0])] += 1


INDEX = ContextIndex()


def log_uncertain(mention, ent, reason):
    if QUIET_UNCERTAIN:
        return
    memory.uncertain.append({
        "entity_type": mention["entity_type"],
        "mention_id": mention["mention_id"],
        "name": mention["raw_name"],
        "clean_name": mention["clean_name"],
        "ucid": mention["ucid"],
        "court": mention["court"],
        "candidate_entity_id": ent["entity_id"],
        "candidate_name": ent["canonical_name"],
        "candidate_courts": ent["courts"][:UNCERTAIN_MAX_COURTS],
        "reason": reason,
        "decision": "NOT_MERGED",
    })


def match_judge(m, entities):
    name = m["clean_name"]
    pool = [entities[eid] for eid in INDEX.by_key.get(("judge", surname(name)), [])]  # unresolved records excluded

    # Surname-only mention: same case -> same division -> same court (title breaks ties);
    # still ambiguous -> the unresolved record for this surname in this court; no judge -> new judge.
    if is_surname_only(name):
        ent, level, amb = pick_by_levels(m, pool)
        if ent:
            return ent["entity_id"]
        if amb:
            for e in amb[:UNCERTAIN_MAX_CANDIDATES]:
                log_uncertain(m, e, f"judge: surname-only mention, {len(amb)} judges with this surname in the same {level}; "
                                    f"sent to the unresolved record")
            return UNRESOLVED
        return None

    cands, other_court = [], []
    for ent in pool:
        if not entity_fits(name, ent, "judge"):
            continue
        if m["court"] in ent["courts"]:
            cands.append(ent)
        elif name in ent["name_variants"]:
            other_court.append(ent)
    exact = [e for e in cands if name in e["name_variants"]]
    if len(exact) == 1:
        return exact[0]["entity_id"]
    if len(cands) == 1:
        return cands[0]["entity_id"]
    if len(cands) > 1:
        t = title_pick(m, cands)
        if t:
            return t["entity_id"]
        for e in cands:
            log_uncertain(m, e, "judge: more than one saved judge in this court fits")
        return None

    # Typo merge: same court + same surname + first name one edit apart (not when both spellings are frequent)
    typo = [e for e in pool if m["court"] in e["courts"] and typo_fits(name, e, m["court"])]
    if len(typo) == 1:
        return typo[0]["entity_id"]
    if len(typo) > 1:
        t = title_pick(m, typo)
        if t:
            return t["entity_id"]
        for e in typo[:UNCERTAIN_MAX_CANDIDATES]:
            log_uncertain(m, e, "judge: first name one edit from more than one judge in this court")
        return None

    # A full name joins an earlier surname-only judge by the same levels.
    so = [e for e in pool if all(is_surname_only(v) for v in e["name_variants"])]
    ent, level, amb = pick_by_levels(m, so)
    if ent:
        return ent["entity_id"]
    if amb:
        for e in amb[:UNCERTAIN_MAX_CANDIDATES]:
            log_uncertain(m, e, f"judge: full name fits more than one surname-only judge in the same {level}")
        return None

    for e in other_court[:UNCERTAIN_MAX_CANDIDATES]:
        log_uncertain(m, e, "judge: same name, different court")
    return None


def unresolved_record(m):
    """One unresolved record per (court, surname); flagged, not counted as a judge."""
    key = ("judge_unresolved", m["court"], surname(m["clean_name"]))
    lst = INDEX.by_key.get(key)
    if lst:
        return lst[0]
    idx = memory.state["counters"].get("judge_unresolved", 1)
    memory.state["counters"]["judge_unresolved"] = idx + 1
    eid = f"ent_{UNRESOLVED_PREFIX}_{idx:06d}"
    memory.entities[eid] = {
        "entity_id": eid,
        "sjid": None,
        "entity_type": "judge",
        "unresolved": True,
        "unresolved_key": {"court": m["court"], "surname": surname(m["clean_name"])},
        "canonical_name": f"unresolved surname: {surname(m['clean_name'])} ({m['court']})",
        "normalized_name": surname(m["clean_name"]),
        "name_variants": [],
        "courts": [],
        "ucids": [],
        "n_mentions": 0,
        "mention_ids": []
    }
    INDEX.by_key[key].append(eid)
    return eid


def match_counsel(m, entities):
    mc = m["context"]
    best, best_score, near = None, 0, []
    mname = counsel_core(m["clean_name"])
    fits = []
    for eid in INDEX.by_key.get(("counsel", surname(mname)), []):
        ent = entities[eid]
        if not entity_fits(mname, ent, "counsel"):
            continue
        fits.append(ent)
        ec = INDEX.ctx[eid]
        score = sum(bool(set(mc.get(f, [])) & ec[f]) for f in ("emails", "phones", "addresses", "offices", "firm_eids"))
        if score > best_score:
            best, best_score = ent, score
        elif score == 0:
            near.append(ent)
    if best is not None:
        return best["entity_id"]
    # Step 5: counsel with no contact info
    if not counsel_has_contact(mc):
        core, _ = person_tokens(mname)
        if len(core) >= COUNSEL_FULL_MIN_TOKENS:
            for ent in fits:
                if m["court"] in ent["courts"] and mname in {counsel_core(v) for v in ent["name_variants"]}:
                    return ent["entity_id"]
        pe = set(mc.get("party_eids", []))
        if pe:
            for ent in fits:
                if same_first_last(mname, ent) and pe & INDEX.ctx[ent["entity_id"]]["party_eids"]:
                    return ent["entity_id"]
    for e in near[:UNCERTAIN_MAX_CANDIDATES]:
        log_uncertain(m, e, "counsel: compatible name, no shared email / phone / address / office")
    return None


def counsel_has_contact(ctx):
    return any(ctx.get(f) for f in ("emails", "phones", "addresses", "offices"))


def same_first_last(mname, ent):
    """Same first and last name; middle names (if both have one) start with the same letter."""
    cm, _ = person_tokens(mname)
    if len(cm) < 2:
        return False
    for v in ent["name_variants"]:
        cv, _ = person_tokens(counsel_core(v))
        if len(cv) >= 2 and cv[0] == cm[0] and cv[-1] == cm[-1]:
            mm, mv = cm[1:-1], cv[1:-1]
            if not mm or not mv or mm[0][0] == mv[0][0]:
                return True
    return False


def match_firm(m, entities):
    fed = federal_agency(m["clean_name"])
    if fed:
        lst = INDEX.by_key.get(("firm_fed", fed, m["court"]), [])
        return lst[0] if lst else None
    mc = m["context"]
    best, best_score, near = None, 0, []
    for eid in INDEX.by_key.get(("firm", m["clean_name"], (mc.get("office") or [""])[0]), []):
        ec = INDEX.ctx[eid]
        score = sum(bool(set(mc.get(f, [])) & ec[f]) for f in ("addresses", "phones", "domains"))
        if score > best_score:
            best, best_score = entities[eid], score
        elif score == 0:
            near.append(entities[eid])
    if best is not None:
        return best["entity_id"]
    for e in near[:UNCERTAIN_MAX_CANDIDATES]:
        log_uncertain(m, e, "firm: same name, no shared address / phone / email domain")
    return None


def party_name_is_rare(m):
    n_cases = len(INDEX.party_name_cases.get(m["clean_name"], ()))
    if n_cases > RARE_MAX_CASES:
        return False
    if m["context"].get("is_org"):
        return True
    return len(m["clean_name"].split()) >= RARE_PERSON_MIN_TOKENS


def match_party(m, entities):
    clean = m["clean_name"]
    if is_doe_entity(clean):
        # Doe names never merge across cases (same rule as the notebook)
        for eid in INDEX.by_key.get(("party", clean), []):
            if m["ucid"] in entities[eid]["ucids"]:
                return eid
        return None
    # Fix 1: one government party across all courts
    if is_gov_party(clean):
        lst = INDEX.by_key.get(("party_gov",), [])
        return lst[0] if lst else None
    # Step 4: acronym -> the org in the same court whose initials match (court-scoped); else the same acronym in that court
    if m["context"].get("acronym"):
        acr = m["context"]["acronym"][0]
        full = [eid for eid in dict.fromkeys(INDEX.by_key.get(("party_initials", m["court"], acr), []))
                if m["ucid"] not in entities[eid]["ucids"]]
        if len(full) == 1:
            return full[0]
        if len(full) > 1:
            for eid in full[:UNCERTAIN_MAX_CANDIDATES]:
                log_uncertain(m, entities[eid], "party: acronym matches the initials of more than one organisation in this court")
            return None
        for eid in INDEX.by_key.get(("party_acronym", m["court"], acr), []):
            if m["ucid"] not in entities[eid]["ucids"]:
                return eid
        return None
    # Fix 2 (+ step 4 key): organisations with an identical key -> one ID, no extra evidence (same-case barrier kept);
    # else an acronym entity in the same court equal to this name's initials
    if m["context"].get("is_org"):
        key = m["context"]["org_key"][0]
        for eid in INDEX.by_key.get(("party", key), []):
            if m["ucid"] not in entities[eid]["ucids"]:
                return eid
        ini = org_initials(key) if " | " not in key else ""
        if ini:
            for eid in INDEX.by_key.get(("party_acronym", m["court"], ini), []):
                if m["ucid"] not in entities[eid]["ucids"]:
                    return eid
        return None
    # Persons: evidence rule
    mc = m["context"]
    rare = party_name_is_rare(m)
    near = []
    for eid in INDEX.by_key.get(("party", clean), []):
        ent = entities[eid]
        if m["ucid"] in ent["ucids"]:
            continue  # never two different names in one case (identical names are one mention per case)
        ec = INDEX.ctx[eid]
        shared_counsel = set(mc.get("counsel_eids", [])) & ec["counsel_eids"]
        shared_firm = set(mc.get("firm_eids", [])) & ec["firm_eids"]
        shared_coparty = {p for p in set(mc.get("coparties", [])) & ec["coparties"]
                          if p != clean and p not in GOV_PARTY_NAMES and not is_doe_entity(p)
                          and len(INDEX.party_name_cases.get(p, ())) <= COPARTY_MAX_CASES}
        if shared_counsel or shared_firm or shared_coparty or rare:
            return eid
        near.append(ent)
    for e in near[:UNCERTAIN_MAX_CANDIDATES]:
        log_uncertain(m, e, "party: same name in another case, no shared counsel / firm / co-party, name not rare")
    return None


MATCHERS = {"judge": match_judge, "counsel": match_counsel, "firm": match_firm, "party": match_party}


# --- 6. EXTRACTION (same mentions as v4_run.py, plus context) ---
def trim_trailing_words(clean):
    """Remove trailing docket words ("<judge> motion", "<judge> final") instead of dropping the name."""
    toks = clean.split()
    while len(toks) > 1 and toks[-1] in DOCKET_TRAILING_WORDS:
        toks.pop()
    return " ".join(toks)


def trim_to_known_judge(m):
    """Trim a docket name to the longest start that equals a known judge's name."""
    toks = m["clean_name"].split()
    if len(toks) < 3:
        return
    for k in range(len(toks) - 1, 1, -1):
        if len(toks[k - 1]) < 2 or toks[k - 1] in FIRST_NAME_LIKE:
            # never trim to a start that ends in an initial or a given name ("<first> <initial>", "<first> <middle>"):
            # the words removed would be the surname
            continue
        prefix = " ".join(toks[:k])
        for eid in INDEX.by_key.get(("judge", surname(prefix)), []):
            if prefix in memory.entities[eid]["name_variants"]:
                m["context"].setdefault("trimmed_from", []).append(m["clean_name"])
                m["clean_name"] = prefix
                return


def is_first_name_only(clean):
    return is_surname_only(clean) and clean in FIRST_NAME_LIKE


def same_case_first_name(m):
    """The one known judge in this case whose first name is this single word."""
    hits = sorted(eid for eid in INDEX.judges_by_case.get(m["ucid"], ())
                  if person_tokens(memory.entities[eid]["normalized_name"])[0][:1] == [m["clean_name"]])
    return hits[0] if len(hits) == 1 else None


def attach_judge(m, eid, stats):
    m.pop("_pending", None)
    n = memory.state["counters"].get("mention", len(memory.mentions))
    m["mention_id"] = f"mnt_{n}"
    memory.state["counters"]["mention"] = n + 1
    e = memory.entities[eid]
    e["n_mentions"] += 1
    e["mention_ids"].append(m["mention_id"])
    for fld, val in (("ucids", m["ucid"]), ("courts", m["court"]), ("name_variants", m["clean_name"])):
        if val not in e[fld]:
            e[fld].append(val)
    m["entity_id"] = eid
    INDEX.register(eid, m)
    memory.mentions[m["mention_id"]] = m
    stats["pending_attached"] = stats.get("pending_attached", 0) + 1


def create_judge(m, stats):
    eid, sid = memory.get_next_id("judge")
    disp = judge_display_name(m["raw_name"])
    if m["context"].get("trimmed_from"):
        disp = " ".join(disp.split()[:len(m["clean_name"].split())])
    memory.entities[eid] = {"entity_id": eid, "sjid": sid, "entity_type": "judge", "canonical_name": disp,
                            "normalized_name": m["clean_name"], "name_variants": [], "courts": [], "ucids": [],
                            "n_mentions": 0, "mention_ids": [], "created_from": "docket text (pending pool)"}
    stats["judges_created_from_docket"] = stats.get("judges_created_from_docket", 0) + 1
    attach_judge(m, eid, stats)
    return eid


def _qualified(ms_by_name):
    """Pending names that may create a judge: first + last name, judge title in DOCKET_CREATE_MIN_CASES+ cases, or an
    FJC-listed name with the title seen once (option A). Returns [(-n_cases, name)] sorted."""
    out = []
    for name, ms in ms_by_name.items():
        if len(person_tokens(name)[0]) < 2 or not any(x["context"].get("docket") for x in ms):
            continue
        n_cases = len({x["ucid"] for x in ms if x["context"].get("docket_title")})
        core = person_tokens(name)[0]
        on_fjc = len(core) >= 2 and (core[0], core[-1]) in FJC_NAMES
        if n_cases >= DOCKET_CREATE_MIN_CASES or (on_fjc and n_cases >= 1):
            out.append((-n_cases, name))
    return sorted(out)


def resolve_pending(stats):
    """End-of-run step for pending judge mentions, in ONE pass over groups keyed by (court, surname):
      1. trim docket names to known judges and join known judges (index lookups only);
      2. per (court, surname) group: promote the qualified name with the most cases, then re-match only that group's
         remaining names (a new judge can only attract names with its surname in its court), repeat inside the group;
      3. names whose trimmed form is now a known judge join it;
      4. one-word first names join the one judge in the same case with that first name (indexed by case).
    Work grows with the number of pending names, not with names x promotions."""
    global QUIET_UNCERTAIN
    QUIET_UNCERTAIN = True
    t0 = time.perf_counter()
    pend = sorted(memory.pending, key=lambda m: (m["court"], m["clean_name"], m["ucid"], m["raw_name"]))
    print(f"End step: pending judges - {len(pend)} mentions", flush=True)
    first_only, rest = [], []
    for m in pend:
        if m["context"].get("docket"):
            trim_to_known_judge(m)
        if is_first_name_only(m["clean_name"]):
            first_only.append(m)
            continue
        r = match_judge(m, memory.entities)
        if r and r != UNRESOLVED:
            attach_judge(m, r, stats)
        else:
            rest.append(m)
    print(f"End step: pending judges - joined known judges, {len(rest)} left | {time.perf_counter() - t0:.1f}s", flush=True)
    groups = defaultdict(list)
    for m in rest:
        groups[(m["court"], surname(m["clean_name"]))].append(m)
    left = []
    for gi, key in enumerate(sorted(groups), start=1):
        ms = groups[key]
        while True:
            by_name = defaultdict(list)
            for x in ms:
                by_name[x["clean_name"]].append(x)
            q = _qualified(by_name)
            if not q:
                break
            name = q[0][1]
            chosen = by_name[name]
            eid = create_judge(chosen[0], stats)
            for x in chosen[1:]:
                attach_judge(x, eid, stats)
            nxt = []
            for x in ms:
                if x["clean_name"] == name:
                    continue
                r = match_judge(x, memory.entities)
                if r and r != UNRESOLVED:
                    attach_judge(x, r, stats)
                else:
                    nxt.append(x)
            ms = nxt
        left.extend(ms)
        if gi % 5000 == 0:
            print(f"End step: pending judges - {gi}/{len(groups)} court+surname groups | {time.perf_counter() - t0:.1f}s", flush=True)
    still = []
    for m in left:
        if m["context"].get("docket"):
            trim_to_known_judge(m)
            if m["context"].get("trimmed_from"):
                r = match_judge(m, memory.entities)
                if r and r != UNRESOLVED:
                    attach_judge(m, r, stats)
                    continue
        still.append(m)
    for m in first_only:
        eid = same_case_first_name(m)
        if eid:
            attach_judge(m, eid, stats)
        else:
            still.append(m)
    memory.pending = sorted(still, key=lambda m: (m["court"], m["clean_name"], m["ucid"], m["raw_name"]))
    QUIET_UNCERTAIN = False
    stats["pending_judge_mentions_left"] = len(memory.pending)
    print(f"End step: pending judges done - {len(memory.pending)} left | {time.perf_counter() - t0:.1f}s", flush=True)


DROPPED = defaultdict(int)  # drop counts by reason, reported in the run summary


def drop_row(rec, reason, stage, field=None, votes=None):
    DROPPED[reason.split(":")[0]] += 1
    row = {"entity_type": rec.get("entity_type"), "raw_name": rec.get("raw_name"), "clean_name": rec.get("clean_name"),
           "ucid": rec.get("ucid"), "court": rec.get("court"), "field": field or FIELD_OF.get(rec.get("entity_type"), ""),
           "reason": reason, "stage": stage}
    if votes:
        row["votes"] = {"spacy": votes[0], "probablepeople": votes[1]}
    return row


def type_gate_stage1(rec):
    """('keep' | 'drop', reason) for counsel / firm / party records from structural signals: no letters, a placeholder
    from v4_rules.yaml, the counsel record's raw_info naming a court-staff role with no email. Whether the string is a
    person, an organisation or not an entity is decided by the name gate (v4_names.NameGate)."""
    t, ctx = rec["entity_type"], rec["_ctx"]
    clean = counsel_core(rec["clean_name"]) if t == "counsel" else rec["clean_name"]
    if not JUNK_MUST_CONTAIN.search(rec["clean_name"]):
        return "drop", "junk: no letters"
    if clean in PLACEHOLDERS.get(t, ()) or (t == "firm" and clean.startswith(FIRM_PLACEHOLDER_PREFIXES)):
        return "drop", f"placeholder: '{clean}'"
    if t in ("counsel", "firm"):
        hit = RAWINFO_STAFF_RE.search(clean)
        if hit and not federal_agency(clean):
            return "drop", f"court staff: role '{hit.group(0)}' in the name"
    if t == "counsel" and ctx.get("rawinfo_staff") and not ctx.get("emails"):
        return "drop", f"court staff: '{ctx['rawinfo_staff'][0]}' in raw_info, no email"
    return "keep", "default"


# --- the name gate (v4_names.NameGate): PERSON / ORGANISATION / NOT_ENTITY ---
ROLE_WORD = {"judge": "judge", "counsel": "lawyer (counsel)", "party": "party", "firm": "law firm or office"}


def gate_name(r):
    """The string the gate reads: judge names without their titles (the same title cleaning as the judge key), other
    names as written."""
    return judge_display_name(r["raw_name"]) if r["entity_type"] == "judge" else r["raw_name"]


def gate_known(r, votes, file_judges=()):
    """Tie-break with data when the two models disagree: (label, reason) or None. file_judges = clean names of the
    judges already accepted from this file's own judge fields (they are known before they are saved)."""
    t, clean, court = r["entity_type"], r["clean_name"], r["court"]
    if t == "judge":
        core = person_tokens(clean)[0]
        for fj in file_judges:
            fc = person_tokens(fj)[0]
            if (person_names_compatible(clean, fj) or (len(core) == 1 and fc and core[0] in (fc[-1], fc[0]))
                    or " ".join(clean.split()[:len(fj.split())]) == fj):
                return N.PERSON, "matches a judge in this case's judge field"
        for eid in INDEX.by_key.get(("judge", surname(clean)), []):
            e = memory.entities[eid]
            if court in e["courts"] and (entity_fits(clean, e, "judge") or (len(core) == 1 and not e.get("unresolved"))):
                return N.PERSON, "matches a known judge in this court"
        if is_first_name_only(clean) and any(person_tokens(memory.entities[eid]["normalized_name"])[0][:1] == [clean]
                                             for eid in INDEX.judges_by_case.get(r["ucid"], ())):
            return N.PERSON, "first name of a judge of this case"
        toks = clean.split()
        for k in range(len(toks) - 1, 1, -1):          # starts with a known judge's full name ("<judge> <docket word>")
            prefix = " ".join(toks[:k])
            for eid in INDEX.by_key.get(("judge", surname(prefix)), []):
                if court in memory.entities[eid]["courts"] and prefix in memory.entities[eid]["name_variants"]:
                    return N.PERSON, "starts with a known judge's name in this court"
        if len(core) >= 2 and (core[0], core[-1]) in FJC_NAMES:
            return N.PERSON, "on the FJC judge list"
        if r["_ctx"].get("structured") and N.PERSON in votes:
            # one model says PERSON and the name is in a judge field of the case
            return N.PERSON, "judge field of the case (structured)"
        return None
    if t == "counsel":
        mname = counsel_core(clean)
        for eid in INDEX.by_key.get(("counsel", surname(mname)), []):
            e = memory.entities[eid]
            if court in e["courts"] and entity_fits(mname, e, "counsel"):
                return N.PERSON, "matches a known lawyer in this court"
        if INDEX.by_key.get(("firm", normalize_entity_name(r["raw_name"], "firm"), N.org_parts(r["raw_name"])["office"])):
            return N.ORG, "matches a known firm / office"
        return None
    if t == "party":
        for eid in INDEX.by_key.get(("party", clean), []):
            return (N.ORG if True in INDEX.ctx[eid]["is_org"] else N.PERSON), "matches a known party"
        return None
    if t == "firm":
        if INDEX.by_key.get(("firm", clean, N.org_parts(r["raw_name"])["office"])):
            return N.ORG, "matches a known firm / office"
        return None
    return None


def gate_decide(r, file_judges=()):
    """(label or None, how, votes). None = undecided (goes to gate_review.jsonl)."""
    sent = r.get("_sent")
    votes = GATE.votes_of(gate_name(r), sent[0] if sent else None)
    if votes[0] is None:
        return None, "no votes", votes
    if votes[0] == votes[1]:
        return votes[0], "both models agree", votes
    known = gate_known(r, votes, file_judges)
    if known:
        return known[0], known[1], votes
    if LLM_CFG:
        lab = GATE.llm(gate_name(r), ROLE_WORD[r["entity_type"]], sent[0] if sent else None)
        if lab:
            return lab, "local LLM", votes
    if r["entity_type"] == "judge" and N.PERSON not in votes:
        # ORGANISATION vs NOT_ENTITY and no data says otherwise: neither model calls it a person, and a judge must be one
        return (votes[0] if votes[0] != N.NOT else votes[1]), "neither model says PERSON", votes
    return None, "undecided", votes


def counsel_as_firm(r, recs):
    """A counsel record the gate calls an ORGANISATION (an office) becomes a firm record of the same file."""
    c = r["_ctx"]
    raw = r["raw_name"]
    for p in recs:
        if p["entity_type"] == "party" and raw in p["_ctx"].get("counsel_raw", []):
            p["_ctx"]["firm_raw"].append(raw)
    return {"raw_name": raw, "clean_name": normalize_entity_name(raw, "firm"), "entity_type": "firm", "ucid": r["ucid"],
            "court": r["court"], "_ctx": defaultdict(list, {
                "addresses": list(c.get("addresses", [])), "phones": list(c.get("phones", [])),
                "domains": sorted({email_domain(e) for e in c.get("emails", []) if email_domain(e)}),
                "office": [N.org_parts(raw)["office"]],
                "converted_from": ["counsel"], "gate": list(c.get("gate", []))})}


def apply_gate(recs, stats):
    """Decide every judge / counsel / party record of one file with the name gate. Returns the records to match."""
    out = []
    file_judges = []
    # judge-field judges first, so docket names of the same case can be checked against them
    order = sorted(range(len(recs)), key=lambda i: (not (recs[i]["entity_type"] == "judge" and recs[i]["_ctx"].get("structured")), i))
    for i in order:
        r = recs[i]
        t = r["entity_type"]
        doc = bool(r["_ctx"].get("document"))   # found by a model in free text (documents / generic JSON), not a field
        if t not in ("judge", "counsel", "party") and not (t == "firm" and doc):
            out.append(r)
            continue
        lab, how, votes = gate_decide(r, file_judges)
        if t == "judge" and lab == N.PERSON and r["_ctx"].get("structured"):
            file_judges.append(r["clean_name"])
        r["_ctx"]["gate"] = [f"{lab or 'undecided'}: {how}"]
        stats[f"gate {t}: {lab or 'undecided'} ({how})"] = stats.get(f"gate {t}: {lab or 'undecided'} ({how})", 0) + 1
        if lab is None:
            memory.gate_review.append({"entity_type": t, "raw_name": r["raw_name"], "ucid": r["ucid"], "court": r["court"],
                                       "votes": {"spacy": votes[0], "probablepeople": votes[1]},
                                       "sentence": (r.get("_sent") or [None])[0],
                                       "kept_as": None if (t == "judge" or doc) else t})
            if doc:
                continue      # a model-found name the gate cannot decide is reviewed, not created
        if t == "firm":
            if lab != N.ORG:
                memory.dropped.append(drop_row(r, f"name gate: document firm is {lab} ({how})", "gate", votes=votes))
                continue
            out.append(r)
            continue
        if t == "judge":
            if lab != N.PERSON:
                if lab is not None:
                    memory.dropped.append(drop_row(r, f"name gate: judge is {lab} ({how})", "gate", votes=votes))
                continue
        elif t == "counsel":
            if lab == N.NOT:
                memory.dropped.append(drop_row(r, f"name gate: counsel is {lab} ({how})", "gate", votes=votes))
                continue
            if lab == N.ORG and RAWINFO_STAFF_RE.search(r["clean_name"]):
                # an organisation named by a court-staff role (probation, pretrial services, clerk) is court staff
                memory.dropped.append(drop_row(r, f"court staff: organisation named by a court-staff role ({how})", "gate",
                                               votes=votes))
                continue
            if lab == N.ORG and GATE_CFG["counsel_organisation"]["value"] == "firm":
                stats["counsel became firm / office"] = stats.get("counsel became firm / office", 0) + 1
                out.append(counsel_as_firm(r, recs))
                continue
            if lab == N.ORG:
                # setting "counsel": an office named as counsel stays a counsel mention, flagged as an organisation
                r["_ctx"]["is_organisation"] = [True]
                stats["counsel flagged as organisation"] = stats.get("counsel flagged as organisation", 0) + 1
        elif t == "party":
            if lab == N.NOT:
                memory.dropped.append(drop_row(r, f"name gate: party is {lab} ({how})", "gate", votes=votes))
                continue
            r["_kind"] = lab if lab in (N.PERSON, N.ORG) else (votes[1] if votes[1] in (N.PERSON, N.ORG) else N.PERSON)
        out.append(r)
    pos = {id(r): k for k, r in enumerate(recs)}
    return sorted(out, key=lambda r: pos.get(id(r), len(recs)))   # keep the file's extraction order


def extract_mentions_from_json(json_content, filename):
    """Returns (records, drops). Every item that is not kept is returned in drops with a reason."""
    data = json.loads(json_content)
    ucid = data.get("ucid")
    if not ucid:
        return [], [{"entity_type": None, "raw_name": None, "clean_name": None, "ucid": None, "court": None,
                     "field": "ucid", "reason": f"file has no ucid: {filename}", "stage": "extraction"}]
    court = data.get("court", "unknown")
    extracted = []  # (raw_name, etype, context[, sentence, (start, end)])
    drops = []
    base_rec = lambda raw, t: {"entity_type": t, "raw_name": str(raw).strip(), "clean_name": "", "ucid": ucid, "court": court}

    # 1. Process Structured Fields
    # Judges
    if data.get("judge"):
        extracted.append((data["judge"], "judge", {"structured": [True]}))
    for rj in data.get("referred_judges", []):
        extracted.append((rj, "judge", {"structured": [True]}))

    # Parties & Nested Judges/Counsel
    for party in data.get("parties", []):
        p_side = side_of(party.get("party_type"), party.get("role"))
        p_ctx = {"counsel_raw": [], "firm_raw": [], "role_raw": [party.get("role") or ""], "side": [p_side]}
        if party.get("name"):
            extracted.append((party["name"], "party", p_ctx))
        if party.get("judge"):
            extracted.append((party["judge"], "judge", {"structured": [True]}))
        for prj in party.get("referred_judges", []):
            extracted.append((prj, "judge", {"structured": [True]}))

        for counsel in party.get("counsel", []):
            if counsel.get("is_pro_se") is True:
                if counsel.get("name"):
                    drops.append(drop_row(base_rec(counsel["name"], "counsel"), "pro se entry (is_pro_se flag)", "extraction"))
                continue
            info = counsel.get("entity_info", {}) or {}
            raw_info = info.get("raw_info") or ""
            ri_staff = RAWINFO_STAFF_RE.search(" ".join(re.sub(r"<[^>]+>|[^\w\s]", " ", raw_info.lower()).split()))
            office = info.get("office_name")
            email = norm_email(info.get("email"))
            phone = norm_phone(info.get("phone"))
            addr = norm_address(info.get("address"))
            if counsel.get("name"):
                extracted.append((counsel["name"], "counsel", {
                    "emails": [email], "phones": [phone], "addresses": [addr],
                    "offices": [normalize_entity_name(office, "firm") if office else ""],
                    "firm_raw": [office.strip()] if office and not office.isdigit() else [],
                    "party_raw": [party["name"].strip()] if party.get("name") else [],
                    "party_side": [p_side],
                    "bar": [True] if BAR_RE.search(raw_info) else [],
                    "rawinfo_staff": [ri_staff.group(0)] if ri_staff else [],
                }))
                p_ctx["counsel_raw"].append(counsel["name"].strip())
            if office and office.isdigit():
                drops.append(drop_row(base_rec(office, "firm"), "junk: office name is only digits", "extraction"))
            if office and not office.isdigit():
                extracted.append((office, "firm", {
                    "addresses": [addr], "phones": [phone], "domains": [email_domain(email)],
                    "office": [N.org_parts(office)["office"]],
                }))
                p_ctx["firm_raw"].append(office.strip())

    # 2. Extract Judges Mentioned Only in Docket Text
    if EXTRACTION_CONFIG["judge"]["docket_scan"]:
        for item in data.get("docket", []):
            txt = item.get("docket_text", "")
            for pattern in DOCKET_JUDGE_PATTERNS:
                for mt in pattern.finditer(txt):
                    candidate = mt.group(1)
                    st = mt.start(1) + len(candidate) - len(candidate.lstrip())
                    en = st + len(candidate.strip())
                    a_, b_ = max(0, st - 200), min(len(txt), en + 200)   # the sentence window the NER model reads
                    extracted.append((candidate.strip(), "judge", {"docket": [True],
                                      "docket_title": [True] if DOCKET_TITLE_RE.search(mt.group(0)) else []},
                                      txt[a_:b_], (st - a_, en - a_)))

    return finish_records(extracted, ucid, court, drops)


def finish_records(extracted, ucid, court, drops):
    """Unique records per file (contexts of repeats merged); the first sentence seen is kept for the name gate."""
    base_rec = lambda raw, t: {"entity_type": t, "raw_name": str(raw).strip(), "clean_name": "", "ucid": ucid, "court": court}
    records = []
    seen = {}
    for item in extracted:
        raw_name, etype, ctx = item[:3]
        sent = (item[3], item[4]) if len(item) > 3 and item[3] and item[4] else None
        clean = normalize_entity_name(raw_name, etype)
        if not clean:
            drops.append(drop_row(base_rec(raw_name, etype), "junk: empty after normalisation", "extraction",
                                  "docket[].docket_text" if ctx.get("docket") else None))
            continue
        if etype == "judge":
            trimmed = trim_trailing_words(clean)
            if trimmed != clean:
                ctx = {**ctx, "trimmed_from": [clean]}
                clean = trimmed
            ctx = {**ctx, "titles": judge_titles(raw_name)}
        key = (raw_name, etype)
        if key not in seen:
            rec = {
                "raw_name": raw_name.strip(),
                "clean_name": clean,
                "entity_type": etype,
                "ucid": ucid,
                "court": court,
                "_ctx": defaultdict(list),
            }
            seen[key] = rec
            records.append(rec)
        rec = seen[key]
        if sent and "_sent" not in rec:
            rec["_sent"] = sent
        for f, vals in ctx.items():
            for v in vals:
                if v and v not in rec["_ctx"][f]:
                    rec["_ctx"][f].append(v)
    party_names = sorted({r["clean_name"] for r in records if r["entity_type"] == "party"})
    for r in records:
        if r["entity_type"] == "party":
            r["_ctx"]["coparties"] = [p for p in party_names if p != r["clean_name"]]
            # organisation / acronym decision needs the gate and the court's acronym set, so it is made at match time
    return records, drops


def extract_text_file(data, filename, kind):
    """Document (OCR pages / blocks) or generic JSON: names found by the model, then the same records as dockets."""
    global EXTRACTOR
    if EXTRACTOR is None:
        EXTRACTOR = READERS.Extractor(DOC_CFG, DOC_CACHE_FILE, DOC_MODEL, LLM_CFG)
    ucid, court, extracted, reviews = READERS.read_text_file(data, filename, kind, EXTRACTOR, DOC_CFG)
    memory.doc_review.extend(reviews)
    records, drops = finish_records(extracted, ucid, court, [])
    party_names = sorted({r["clean_name"] for r in records if r["entity_type"] == "party"})
    for r in records:
        if r["entity_type"] == "party":
            r["_ctx"]["coparties"] = [p for p in party_names if p != r["clean_name"]]
    return records, drops


def extract_any(content, filename):
    """Pick the reader from the file's structure (not its name)."""
    try:
        data = json.loads(content)
    except Exception as ex:
        return [], [{"entity_type": None, "raw_name": None, "clean_name": None, "ucid": None, "court": None,
                     "field": "file", "reason": f"not valid JSON ({type(ex).__name__}): {filename}", "stage": "extraction"}], "invalid"
    kind = READERS.file_kind(data)
    if kind == "docket":
        recs, drops = extract_mentions_from_json(content, filename)
    else:
        recs, drops = extract_text_file(data, filename, kind)
    return recs, drops, kind


def ingest_json_files(files_dict):
    """
    files_dict: {filename: content_bytes}
    """
    stats = {
        "files_processed": 0,
        "files_skipped": 0,
        "mentions_extracted": 0,
        "matched_existing": 0,
        "new_entities": 0,
        "uncertain_pairs_logged": 0,
    }
    n_uncertain_before = len(memory.uncertain)

    total = len(files_dict)
    t_start = t_block = time.perf_counter()
    print(f"Starting: {total} input file(s)", flush=True)

    def report_progress(done):
        nonlocal t_block
        if done % PROGRESS_EVERY:
            return
        now = time.perf_counter()
        rate = PROGRESS_EVERY / (now - t_block) * 60 if now > t_block else 0.0
        t_block = now
        print(f"Progress: {done}/{total} files | {len(memory.mentions)} mentions | {len(memory.entities)} entities | "
              f"elapsed {now - t_start:.1f}s | {rate:.0f} files/min (last {PROGRESS_EVERY})", flush=True)

    # Pass 1: extract every new file once, so party-name counts ("rare name") cover the whole batch.
    todo = []
    for fname, content in files_dict.items():
        checksum = hashlib.sha256(content).hexdigest()
        if fname in memory.state["processed_files"] and memory.state["processed_files"][fname] == checksum:
            todo.append((fname, checksum, None, None))
            continue
        recs, drops, kind = extract_any(content.decode("utf-8", errors="replace"), fname)
        stats[f"files read as {kind}"] = stats.get(f"files read as {kind}", 0) + 1
        for r in recs:
            if r["entity_type"] == "counsel":
                INDEX.counsel_name_cases[counsel_core(r["clean_name"])].add(r["ucid"])
            if r["entity_type"] == "party":
                INDEX.add_party_name_case(r["clean_name"], r["ucid"])
            elif r["entity_type"] == "judge":
                INDEX.add_judge_first(r["clean_name"], r["court"])
        todo.append((fname, checksum, recs, drops))
        if len(todo) % PROGRESS_EVERY == 0:
            print(f"Reading: {len(todo)}/{total} files | {time.perf_counter() - t_start:.1f}s", flush=True)

    # Name gate: two model votes per DISTINCT string (cached across runs), then party initials for the acronym rule
    items = {}
    for _, _, recs, _ in todo:
        for r in recs or []:
            if r["entity_type"] in ("judge", "counsel", "party") or (r["entity_type"] == "firm" and r["_ctx"].get("document")):
                sent = r.get("_sent")
                items.setdefault(N.NameGate.key(gate_name(r), sent[0] if sent else None),
                                 (gate_name(r), sent[0] if sent else None, sent[1] if sent else None))
    t_g = time.perf_counter()
    print(f"Name gate: {len(items)} distinct strings in this batch, {sum(k not in GATE.cache for k in items)} new", flush=True)
    stats["name_gate_new_strings"] = GATE.vote(items, progress=lambda msg: print(msg, flush=True))
    stats["name_gate_sec"] = round(time.perf_counter() - t_g, 1)
    print(f"Name gate: done | {stats['name_gate_sec']}s", flush=True)
    for _, _, recs, _ in todo:
        for r in recs or []:
            if r["entity_type"] == "party":
                s_, p_ = GATE.votes_of(r["raw_name"])
                kind = s_ if s_ == p_ else p_
                r["_ctx"]["batch_initials"] = org_batch_initials(r["raw_name"], r["clean_name"], kind)
                INDEX.add_org_initials(r["_ctx"], r["court"])
            if r["entity_type"] in ("counsel", "firm", "party"):
                r["_gate"] = type_gate_stage1(r)

    # Pass 2: match file by file, in sorted name order.
    for done, (fname, checksum, raw_mentions, file_drops) in enumerate(todo, start=1):
        if raw_mentions is None:
            stats["files_skipped"] += 1
            report_progress(done)
            continue
        stats["files_processed"] += 1
        memory.dropped.extend(file_drops or [])
        kept = []
        for r in raw_mentions:
            gate = r.pop("_gate", None)
            if gate and gate[0] == "drop":
                memory.dropped.append(drop_row(r, gate[1], "stage1"))
                continue
            kept.append(r)
        raw_mentions = apply_gate(kept, stats)
        stats["mentions_extracted"] += len(raw_mentions)

        # mention IDs come from a counter, so a mention dropped later in this file leaves a gap instead of a reused ID
        base = memory.state["counters"].get("mention", len(memory.mentions))
        for i, m_data in enumerate(raw_mentions):
            m_data["mention_id"] = f"mnt_{base + i}"
        memory.state["counters"]["mention"] = base + len(raw_mentions)
        judges_in_file = []

        # firms, counsel with contact info, parties (which use the resolved counsel / firm IDs), counsel without contact
        # info (step 5: they use the resolved party IDs), then judges
        eid_of = {}
        for etype, want_contact in (("firm", None), ("counsel", True), ("party", None), ("counsel", False), ("judge", None)):
            if etype == "counsel" and want_contact is False:
                # counsel already matched in this file now record the party IDs they represent
                for mm in raw_mentions:
                    if mm["entity_type"] == "counsel" and "context" in mm and "party_raw" in mm["context"]:
                        pe = sorted({eid_of[("party", r)] for r in mm["context"].pop("party_raw") if ("party", r) in eid_of})
                        mm["context"]["party_eids"] = pe
                        INDEX.ctx[mm["entity_id"]]["party_eids"].update(pe)
            for m_data in raw_mentions:
                if m_data["entity_type"] != etype or "_ctx" not in m_data:
                    continue
                if want_contact is not None and counsel_has_contact(m_data["_ctx"]) != want_contact:
                    continue
                c = m_data.pop("_ctx")
                if etype == "counsel" and want_contact is False:
                    c["party_eids"] = sorted({eid_of[("party", r)] for r in c.pop("party_raw", []) if ("party", r) in eid_of})
                if etype == "counsel":
                    c["firm_eids"] = sorted({eid_of[("firm", r)] for r in c.pop("firm_raw", []) if ("firm", r) in eid_of})
                if etype == "party":
                    c.update(party_org_decision(m_data["raw_name"], m_data["clean_name"], m_data["court"],
                                                m_data.pop("_kind", N.PERSON)))
                    cop = c.get("coparties", [])
                    c["coparties"] = [] if len(cop) > COPARTY_MAX_PARTIES_IN_CASE else [
                        p for p in cop if p not in GOV_PARTY_NAMES and not is_doe_entity(p)
                        and len(INDEX.party_name_cases.get(p, ())) <= COPARTY_MAX_CASES]
                    c["counsel_eids"] = sorted({eid_of[("counsel", r)] for r in c.pop("counsel_raw", []) if ("counsel", r) in eid_of})
                    c["firm_eids"] = sorted({eid_of[("firm", r)] for r in c.pop("firm_raw", [])
                                             if ("firm", r) in eid_of
                                             and not federal_agency(memory.entities[eid_of[("firm", r)]]["normalized_name"])})
                if etype == "party":
                    m_data["role"] = [x for x in c.pop("role_raw", []) if x]
                    m_data["side"] = c.pop("side", [])
                if etype == "counsel":
                    m_data["side"] = c.pop("party_side", [])
                m_data.pop("_sent", None)
                m_data["context"] = dict(c)
                m_id = m_data["mention_id"]

                matched_ent_id = None
                docket_only = etype == "judge" and bool(c.get("docket")) and not c.get("structured")
                if docket_only:
                    trim_to_known_judge(m_data)
                if etype == "judge" and is_first_name_only(m_data["clean_name"]):
                    # a first name alone never creates a judge ID: join the one judge in this case with that first name,
                    # else wait in the pending pool
                    matched_ent_id = same_case_first_name(m_data)
                    if not matched_ent_id:
                        m_data["_pending"] = True
                        memory.pending.append(m_data)
                        continue
                if not matched_ent_id:
                    matched_ent_id = MATCHERS[etype](m_data, memory.entities)
                if docket_only and (not matched_ent_id or matched_ent_id == UNRESOLVED):
                    # no known judge yet: wait in the pending pool (promoted at the end of the run if it qualifies)
                    m_data["_pending"] = True
                    memory.pending.append(m_data)
                    continue
                if matched_ent_id == UNRESOLVED:
                    m_data["unresolved"] = True
                    matched_ent_id = unresolved_record(m_data)
                    stats["judge_mentions_unresolved"] = stats.get("judge_mentions_unresolved", 0) + 1

                if matched_ent_id:
                    ent = memory.entities[matched_ent_id]
                    ent["n_mentions"] += 1
                    if m_id not in ent["mention_ids"]:
                        ent["mention_ids"].append(m_id)
                    if m_data["ucid"] not in ent["ucids"]:
                        ent["ucids"].append(m_data["ucid"])
                    if m_data["court"] not in ent["courts"]:
                        ent["courts"].append(m_data["court"])
                    if m_data["clean_name"] not in ent["name_variants"]:
                        ent["name_variants"].append(m_data["clean_name"])
                    if etype == "party":
                        rc = ent.setdefault("role_counts", {})
                        for sd in m_data.get("side", []):
                            rc[sd] = rc.get(sd, 0) + 1
                    stats["matched_existing"] += 1
                    m_data["entity_id"] = matched_ent_id
                else:
                    ent_id, specific_id = memory.get_next_id(etype)
                    id_key = EXTRACTION_CONFIG[etype]["id_key"]
                    memory.entities[ent_id] = {
                        "entity_id": ent_id,
                        id_key: specific_id,
                        "entity_type": etype,
                        "canonical_name": judge_display_name(m_data["raw_name"]) if etype == "judge" else N.squash(m_data["raw_name"]),
                        "normalized_name": m_data["clean_name"],
                        "name_variants": [m_data["clean_name"]],
                        "courts": [m_data["court"]],
                        "ucids": [m_data["ucid"]],
                        "n_mentions": 1,
                        "mention_ids": [m_id]
                    }
                    if etype == "party":
                        rc = memory.entities[ent_id].setdefault("role_counts", {})
                        for sd in m_data.get("side", []):
                            rc[sd] = rc.get(sd, 0) + 1
                    m_data["entity_id"] = ent_id
                    stats["new_entities"] += 1
                INDEX.register(m_data["entity_id"], m_data)
                eid_of[(etype, m_data["raw_name"])] = m_data["entity_id"]
                if etype == "judge" and not m_data.get("unresolved"):
                    judges_in_file.append(m_data["entity_id"])

        for m_data in raw_mentions:  # mentions.jsonl keeps extraction order; dropped judge mentions are not stored
            if "_drop" in m_data or "_pending" in m_data:
                continue
            memory.mentions[m_data["mention_id"]] = m_data
        memory.state["processed_files"][fname] = checksum
        report_progress(done)

    print(f"End step: matching done | {time.perf_counter() - t_start:.1f}s", flush=True)
    for m in memory.pending:
        m.pop("_sent", None)
    resolve_pending(stats)
    refresh_display_names()
    print(f"End step: display names | {time.perf_counter() - t_start:.1f}s", flush=True)
    memory.save()
    stats["uncertain_pairs_logged"] = len(memory.uncertain) - n_uncertain_before
    for k_, v_ in sorted(DROPPED.items(), key=lambda kv: -kv[1]):
        stats[f"dropped: {k_}"] = v_

    print("\n" + "="*45)
    print(" INGESTION PIPELINE REPORT")
    print("="*45)
    for k, v in stats.items():
        print(f" {k.replace("_", " ").capitalize():<25}: {v}")
    print("="*45, flush=True)
    return stats


def refresh_display_names():
    """canonical_name = the most complete variant rebuilt from the parser's components (no titles / prefixes / line
    breaks). Persons: judges, counsel, person parties; organisations: firms and organisation parties. IDs never change."""
    raws = defaultdict(set)
    org = set()
    for m in memory.mentions.values():
        eid = m.get("entity_id")
        if not eid:
            continue
        raw = m.get("raw_name") or ""
        if m["entity_type"] == "judge":
            raw = judge_display_name(raw)
            if (m.get("context") or {}).get("trimmed_from"):
                raw = " ".join(raw.split()[:len(m["clean_name"].split())])
        raws[eid].add(raw)
        if m["entity_type"] == "party" and (m.get("context") or {}).get("is_org"):
            org.add(eid)
    for eid, e in memory.entities.items():
        if e.get("unresolved") or eid not in raws:
            continue
        kind = N.ORG if e["entity_type"] == "firm" or eid in org else N.PERSON
        d = N.display_name(raws[eid], kind)
        if d:
            e["canonical_name"] = d


def rebuild_index():
    """Rebuild the candidate index and evidence from saved entities and mentions."""
    for m in memory.mentions.values():
        if m["entity_type"] == "party":
            INDEX.add_party_name_case(m["clean_name"], m["ucid"])
            INDEX.add_org_initials(m.get("context"), m["court"])
        elif m["entity_type"] == "judge" and not m.get("unresolved"):
            INDEX.add_judge_first(m["clean_name"], m["court"])
        elif m["entity_type"] == "counsel":
            INDEX.counsel_name_cases[counsel_core(m["clean_name"])].add(m["ucid"])
        INDEX.register(m["entity_id"], m)


# --- LOCAL INPUT ---
def collect_inputs(paths):
    """Return {filename: content_bytes} in sorted filename order.
    Accepts .json files, folders of .json files (top level), and .zip files."""
    files = {}
    for p in map(lambda s: Path(s).expanduser(), paths):
        if p.is_dir():
            for fp in p.glob("*.json"):
                files[fp.name] = fp.read_bytes()
        elif p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p, "r") as z:
                for name in z.namelist():
                    if name.endswith(".json"):
                        files[Path(name).name] = z.read(name)
        elif p.suffix.lower() == ".json" and p.is_file():
            files[p.name] = p.read_bytes()
        else:
            raise SystemExit(f"ERROR: not a .json file, folder or .zip: {p}")
    return dict(sorted(files.items()))


def main():
    global STORAGE_DIR, ENTITIES_FILE, MENTIONS_FILE, STATE_FILE, UNCERTAIN_FILE, DROPPED_FILE, PENDING_FILE
    global GATE_CACHE_FILE, GATE_REVIEW_FILE, DOC_CACHE_FILE, DOC_REVIEW_FILE, GATE, DOC_MODEL, LLM_CFG, memory
    faulthandler.register(signal.SIGUSR1, all_threads=True)   # `kill -USR1 <pid>` prints where the run is
    ap = argparse.ArgumentParser(description="V4 incremental entity resolution, context-aware matching (local)")
    ap.add_argument("--input", nargs="+", required=True, help=".json file(s), folder(s) of .json files, or .zip")
    ap.add_argument("--output", required=True, help="folder for entities.jsonl, mentions.jsonl, state.json, uncertain_pairs.jsonl")
    ap.add_argument("--rules", default=None, help="rules file (default: v4_rules.yaml next to this script)")
    ap.add_argument("--llm", choices=["local"], default=None,
                    help="local Ollama model (v4_rules.yaml name_gate.ollama) settles name-gate disagreements and, with "
                         "--doc-model ollama, reads documents; answers are cached")
    ap.add_argument("--doc-model", choices=["gliner2", "ollama"], default="gliner2",
                    help="model that finds names in OCR documents and other JSON (default gliner2)")
    ap.add_argument("--order", choices=["name", "reverse"], default="name",
                    help="file order (default: sorted by name; 'reverse' is for order-sensitivity tests)")
    args = ap.parse_args()

    if args.rules:
        load_rules(args.rules)
    STORAGE_DIR = Path(args.output).expanduser()
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    ENTITIES_FILE = STORAGE_DIR / "entities.jsonl"
    MENTIONS_FILE = STORAGE_DIR / "mentions.jsonl"
    STATE_FILE = STORAGE_DIR / "state.json"
    UNCERTAIN_FILE = STORAGE_DIR / "uncertain_pairs.jsonl"
    DROPPED_FILE = STORAGE_DIR / "dropped_entities.jsonl"
    PENDING_FILE = STORAGE_DIR / "pending_judges.jsonl"
    GATE_CACHE_FILE = STORAGE_DIR / "name_gate_cache.jsonl"
    GATE_REVIEW_FILE = STORAGE_DIR / "gate_review.jsonl"
    DOC_CACHE_FILE = STORAGE_DIR / "document_model_cache.jsonl"
    DOC_REVIEW_FILE = STORAGE_DIR / "document_review.jsonl"
    DOC_MODEL = args.doc_model
    if args.llm == "local" or args.doc_model == "ollama":
        oc = GATE_CFG["ollama"]
        if N.ollama_available(oc["url"], oc["model"]):
            LLM_CFG = dict(oc)
        else:
            print(f"WARNING: Ollama model {oc['model']} is not available at {oc['url']} - the LLM stays off "
                  f"(install Ollama, run `ollama pull {oc['model']}`)", flush=True)
            if args.doc_model == "ollama":
                raise SystemExit("ERROR: --doc-model ollama needs Ollama")
    GATE = N.NameGate(GATE_CACHE_FILE, GATE_CFG["spacy_model"], LLM_CFG)

    t0 = time.perf_counter()
    memory = MemoryManager(STORAGE_DIR)
    rebuild_index()
    files = collect_inputs(args.input)
    if args.order == "reverse":
        files = dict(reversed(list(files.items())))
    print(f"Ingesting {len(files)} file(s)...", flush=True)
    ingest_json_files(files)

    counts = defaultdict(int)
    for ent in memory.entities.values():
        counts[ent["entity_type"] + ("_unresolved" if ent.get("unresolved") else "")] += 1
    print(f"Totals: {len(memory.entities)} entities {dict(counts)}, {len(memory.mentions)} mentions, "
          f"{len(memory.state['processed_files'])} processed files, {len(memory.uncertain)} uncertain pairs. "
          f"Time {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

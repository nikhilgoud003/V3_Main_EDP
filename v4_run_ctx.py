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
import re
import json
import hashlib
import time
import zipfile
from collections import defaultdict
from pathlib import Path

# --- RULES: every word list, pattern and threshold comes from v4_rules.yaml (see the `source:` notes there) ---
import yaml

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
    global RULES, RULES_FILE, EXTRACTION_CONFIG, DOCKET_JUDGE_PATTERNS, NON_JUDGE_DOCKET_KEYWORDS
    global RARE_MAX_CASES, RARE_PERSON_MIN_TOKENS, COPARTY_MAX_CASES, COPARTY_MAX_PARTIES_IN_CASE, PROGRESS_EVERY
    global WEBMAIL_DOMAINS, NAME_SUFFIXES, COUNSEL_TAG_TOKENS, GOV_PARTY_NAMES, GOV_NOT_IF_CONTAINS
    global ORG_PARTY_RE, PLACEHOLDER_RE, USAO_RE, FPD_RE, NICKNAMES, DOE_PATTERN, CORP_SUFFIX_PATTERN, TITLE_PATTERN
    global JUDGE_TITLE_PATTERN, APOSTROPHE_RE, PHONE_MIN_DIGITS, PHONE_COUNTRY_CODE, ZIP_RE, STREET_NUMBER_RE
    global UNCERTAIN_MAX_CANDIDATES, UNCERTAIN_MAX_COURTS, MATCHING_LABEL
    global JUDGE_PARENS_RE, JUDGE_NON_NAME_WORDS, JUDGE_TITLE_EVIDENCE, TYPO_MAX_EDITS, TYPO_MIN_LEN, TYPO_BOTH_FREQ
    global SURNAME_ONLY_LEVELS, UNRESOLVED_PREFIX
    global COUNSEL_FULL_MIN_TOKENS, FJC_NAMES, FIRST_NAME_LIKE
    global STAFF_RE, PARTY_STAFF_MAX_OTHER, PLACEHOLDERS, FIRM_PLACEHOLDER_PREFIXES, BAR_RE, UNDECIDED_MIN_CASES, TYPE_MODEL_CFG
    global SIDE_BY_PTYPE, SIDE_BY_ROLE, DEFAULT_SIDE, JUNK_MUST_CONTAIN
    global DOCKET_CREATE_MIN_CASES, DOCKET_TITLE_RE, DOCKET_TRAILING_WORDS
    global ORG_ABBREV, ORG_DROP_WORDS, ORG_COLLAPSE, ACR_MIN, ACR_MAX, INITIALS_SKIP, DIV_SEP_RE, DIV_WORDS, DIV_ACR_MAX, LEGAL_FORMS
    if path:
        RULES_FILE = Path(path).expanduser()
    with open(RULES_FILE, encoding="utf-8") as f:
        R = yaml.safe_load(f)
    RULES = R
    EXTRACTION_CONFIG = {t: dict(R["entity_types"][t]) for t in ("judge", "party", "counsel", "firm")}
    DOCKET_JUDGE_PATTERNS = [re.compile(p) for p in R["docket_judge_patterns"]["patterns"]]
    NON_JUDGE_DOCKET_KEYWORDS = set(R["non_judge_docket_keywords"]["words"])
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
    ow = R["org_party_words"]
    ORG_PARTY_RE = re.compile(_word_re(ow["words"]) + r"|\b(" + "|".join(ow["of_words"]["words"]) + r") of\b")
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
    CORP_SUFFIX_PATTERN = re.compile(_word_re(R["corp_suffix_words"]["words"]))
    TITLE_PATTERN = re.compile(_word_re(R["title_words"]["words"]))
    J = R["judges"]
    JUDGE_TITLE_PATTERN = re.compile(_word_re(R["judge_title_words"]["words"] + J["title_typos"]["words"]))
    JUDGE_PARENS_RE = re.compile(J["strip_parenthetical"]["pattern"])
    JUDGE_NON_NAME_WORDS = set(J["non_name_words"]["words"])
    JUDGE_TITLE_EVIDENCE = {t: list(ws) for t, ws in J["title_evidence"]["titles"].items()}
    TYPO_MAX_EDITS = J["typo"]["max_first_name_edits"]
    TYPO_MIN_LEN = J["typo"]["min_first_name_length"]
    TYPO_BOTH_FREQ = J["typo"]["both_frequent_min_mentions"]
    SURNAME_ONLY_LEVELS = list(J["surname_only"]["levels"])
    UNRESOLVED_PREFIX = J["surname_only"]["unresolved_id_prefix"]
    O = R["organisations"]
    ORG_ABBREV = dict(O["abbreviations"]["map"])
    ORG_DROP_WORDS = set(O["abbreviations"]["drop_words"])
    ORG_COLLAPSE = bool(O["collapse_single_letters"]["enabled"])
    ACR_MIN, ACR_MAX = O["acronym"]["min_length"], O["acronym"]["max_length"]
    INITIALS_SKIP = set(O["acronym"]["initials_skip_words"])
    DIV_SEP_RE = re.compile(O["division"]["separator_pattern"])
    DIV_WORDS = set(O["division"]["words"])
    DIV_ACR_MAX = O["division"]["acronym_max_length"]
    LEGAL_FORMS = set(R["corp_suffix_words"]["words"]) | set(O["legal_forms"]["extra_words"])
    COUNSEL_FULL_MIN_TOKENS = R["counsel"]["no_contact"]["full_name_min_tokens"]
    rel = lambda x: (Path(x).expanduser() if Path(x).expanduser().is_absolute() else RULES_FILE.parent / x)
    FJC_NAMES = load_fjc_names(rel(R["docket_judges"]["fjc_list"]["path"]))
    fn = R["docket_judges"]["first_name_only"]
    FIRST_NAME_LIKE = load_first_name_like(rel(fn["male_first"]), rel(fn["female_first"]), rel(fn["surnames"]))
    G = R["type_gate"]
    STAFF_RE = re.compile(_word_re([re.escape(w) for w in G["court_staff_phrases"]]))
    PARTY_STAFF_MAX_OTHER = G["party_staff_max_other_words"]["value"]
    PLACEHOLDERS = {t: set(G["placeholders"][t]) for t in ("counsel", "firm", "party")}
    FIRM_PLACEHOLDER_PREFIXES = tuple(G["placeholders"]["firm_prefixes"])
    BAR_RE = re.compile(G["bar_pattern"]["pattern"], re.I)
    JUNK_MUST_CONTAIN = re.compile(G["junk_rule"]["must_contain_pattern"])
    DJ = R["docket_judges"]
    DOCKET_CREATE_MIN_CASES = DJ["rule"]["min_cases"]
    DOCKET_TITLE_RE = re.compile(_word_re([re.escape(w) for w in DJ["rule"]["title_words"]]), re.I)
    DOCKET_TRAILING_WORDS = set(DJ["trailing_words"]["words"])
    UNDECIDED_MIN_CASES = G["undecided_min_cases"]["value"]
    TYPE_MODEL_CFG = G["model"]
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
TYPE_CACHE_FILE = None
PENDING_FILE = None
QUIET_UNCERTAIN = False   # set while re-checking pending judges, so uncertain pairs are not logged twice
TYPE_MODEL = "none"   # --type-model
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
        self.type_cache = {}  # Stage 2 answers: key -> row
        self.pending = []     # docket / one-word judge mentions waiting to join or create a judge
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
        if TYPE_CACHE_FILE.exists():
            with open(TYPE_CACHE_FILE, "r", encoding="utf-8") as f:
                for l in f:
                    if l.strip():
                        row = json.loads(l)
                        self.type_cache[row["key"]] = row
        if self.mentions and not all("context" in m for m in self.mentions.values()):
            raise SystemExit("ERROR: this output folder was made by v4_run.py (mentions have no context). "
                             "Use a new --output folder for v4_run_ctx.py.")
        print(f"Memory Loaded: {len(self.entities)} entities, {len(self.mentions)} mentions, "
              f"{len(self.state['processed_files'])} processed files.", flush=True)

    def save(self):
        self.state["matching"] = MATCHING_LABEL
        self.state.pop("docket_judge_cases", None)
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
        with open(ENTITIES_FILE, "w", encoding="utf-8") as f:
            for ent in self.entities.values():
                f.write(json.dumps(ent) + "\n")
        with open(MENTIONS_FILE, "w", encoding="utf-8") as f:
            for mnt in self.mentions.values():
                f.write(json.dumps(mnt) + "\n")
        with open(UNCERTAIN_FILE, "w", encoding="utf-8") as f:
            for row in self.uncertain:
                f.write(json.dumps(row) + "\n")
        with open(DROPPED_FILE, "w", encoding="utf-8") as f:
            for row in self.dropped:
                f.write(json.dumps(row) + "\n")
        with open(PENDING_FILE, "w", encoding="utf-8") as f:
            for row in self.pending:
                f.write(json.dumps(row) + "\n")
        with open(TYPE_CACHE_FILE, "w", encoding="utf-8") as f:
            for row in self.type_cache.values():
                f.write(json.dumps(row) + "\n")

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
        # Standardize corporate suffixes for comparison
        base_corp = CORP_SUFFIX_PATTERN.sub("", base)
        return " ".join(base_corp.split()).strip() or base
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


def judge_is_non_name(clean_name):
    return any(t in JUDGE_NON_NAME_WORDS for t in clean_name.split())


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


# --- Step 4: organisation keys, divisions, acronyms ---
def collapse_letters(text):
    """'l l c' -> 'llc': a run of single letters is joined only when it spells a legal form."""
    if not ORG_COLLAPSE:
        return text
    out, run = [], []

    def flush():
        joined = "".join(run)
        out.extend([joined] if joined in LEGAL_FORMS else run)
        run.clear()

    for t in text.split():
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            flush()
        out.append(t)
    if run:
        flush()
    return " ".join(out)


def org_text(raw_name):
    """Cleaned, single letters joined, abbreviations expanded (legal forms still present)."""
    toks = collapse_letters(clean_text_base(raw_name)).split()
    return " ".join(ORG_ABBREV.get(t, t) for t in toks)


def is_acronym(raw_part):
    t = str(raw_part).strip().replace(".", "")
    return t.isalpha() and t.isupper() and ACR_MIN <= len(t) <= ACR_MAX and " " not in t


def _suffix_is_legal_form(suffix):
    toks = collapse_letters(clean_text_base(suffix)).split()
    return bool(toks) and set(toks) <= LEGAL_FORMS


def split_division(raw_name):
    """'Parent - X Division' -> 'Parent' (suffix with a division word). Legal-form suffixes never split.
    Single all-caps suffixes ('-CID') are handled by acronym_suffix(), which needs the court."""
    raw = str(raw_name)
    for m in DIV_SEP_RE.finditer(raw):
        parent, suffix = raw[:m.start()].strip(), raw[m.end():].strip()
        if not parent or not suffix or _suffix_is_legal_form(suffix):
            continue
        if set(clean_text_base(suffix).split()) & DIV_WORDS:
            return parent
    return raw


def division_suffix_initials(raw_name):
    """Initials of a division-word suffix ('Field Operations Division' -> 'fod'), for the court's acronym set."""
    raw = str(raw_name)
    for m in DIV_SEP_RE.finditer(raw):
        parent, suffix = raw[:m.start()].strip(), raw[m.end():].strip()
        if parent and suffix and not _suffix_is_legal_form(suffix) and set(clean_text_base(suffix).split()) & DIV_WORDS:
            key = " ".join(t for t in org_text(suffix).split() if t not in ORG_DROP_WORDS)
            return org_initials(key)
    return ""


def acronym_suffix(raw_form):
    """(parent, suffix) for the first separator whose suffix is one all-caps token ('ABCD-FOD' -> ('ABCD', 'fod'))."""
    raw = str(raw_form)
    for m in DIV_SEP_RE.finditer(raw):
        parent, suffix = raw[:m.start()].strip(), raw[m.end():].strip()
        t = suffix.replace(".", "")
        if parent and t.isalpha() and t.isupper() and 2 <= len(t) <= DIV_ACR_MAX and not _suffix_is_legal_form(suffix):
            return parent, t.lower()
    return None


def batch_initials(raw_name, clean):
    """Initials this party name adds to its court's acronym set: a full organisation name, and a division-word suffix."""
    out = []
    if not is_gov_party(clean) and not is_placeholder_party(clean):
        form = split_division(raw_name)
        if not is_acronym(form) and is_org_name(raw_name, clean):
            ini = org_initials(org_key(form))
            if ini:
                out.append(ini)
        d = division_suffix_initials(raw_name)
        if d:
            out.append(d)
    return out


def party_org_decision(raw_name, clean, court):
    """Context fields for a party at match time: is_org, and org_key or acronym."""
    if is_gov_party(clean) or is_placeholder_party(clean):
        return {"is_org": []}
    known = INDEX.org_initials.get(court, ())
    form = split_division(raw_name)
    has_div = form != str(raw_name)
    sp = acronym_suffix(form)
    if sp and sp[1] in known:
        form, has_div = sp[0], True
    if is_acronym(form):
        acr = form.strip().replace(".", "").lower()
        if has_div or acr in known:
            return {"is_org": [True], "acronym": [acr]}
        return {"is_org": []}
    if is_org_name(raw_name, clean):
        return {"is_org": [True], "org_key": [org_key(form)]}
    return {"is_org": []}


def org_key(raw_name):
    """Matching key for an organisation party (callers pass the parent form when a division suffix applies)."""
    parent = split_division(raw_name)
    toks = [t for t in CORP_SUFFIX_PATTERN.sub(" ", org_text(parent)).split() if t not in ORG_DROP_WORDS]
    return " ".join(toks) or normalize_entity_name(parent, "party")


def org_initials(key):
    toks = [t for t in key.split() if t not in INITIALS_SKIP]
    ini = "".join(t[0] for t in toks)
    return ini if len(toks) >= 2 and ACR_MIN <= len(ini) <= ACR_MAX else ""


def is_org_name(raw_name, clean):
    """Organisation party: legal form, government / agency word, or "State / County / City of". Never a placeholder.
    Dotted forms and abbreviations are normalised first. (All-caps acronyms are decided separately, at match time.)"""
    if is_placeholder_party(clean):
        return False
    return bool(ORG_PARTY_RE.search(org_text(raw_name)))


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

    def keys_for(self, ent_type, clean, court, unresolved=False, ctx=None):
        ctx = ctx or {}
        if ent_type == "party" and not is_gov_party(clean):
            if ctx.get("acronym"):
                return [("party_acronym", court, ctx["acronym"][0])]
            if ctx.get("org_key"):
                k = ctx["org_key"][0]
                ini = org_initials(k)
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
        return [("firm", clean)]

    def register(self, ent_id, mention):
        """Add a mention's name keys and context to its entity."""
        t, clean, court = mention["entity_type"], mention["clean_name"], mention["court"]
        for k in self.keys_for(t, clean, court, mention.get("unresolved", False), mention.get("context")):
            lst = self.by_key[k]
            if ent_id not in lst:
                lst.append(ent_id)
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
    for eid in INDEX.by_key.get(("firm", m["clean_name"]), []):
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
        ini = org_initials(key)
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
    while len(toks) > 1 and (toks[-1] in DOCKET_TRAILING_WORDS or toks[-1] in JUDGE_NON_NAME_WORDS):
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
    hits = sorted({eid for eid, e in memory.entities.items()
                   if e["entity_type"] == "judge" and not e.get("unresolved") and m["ucid"] in e["ucids"]
                   and person_tokens(e["normalized_name"])[0][:1] == [m["clean_name"]]})
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


def resolve_pending(stats):
    """Re-check every pending judge mention: join a known judge, or promote a qualified docket name (first + last name,
    judge title, DOCKET_CREATE_MIN_CASES+ cases) with all its mentions. Repeats until nothing changes."""
    global QUIET_UNCERTAIN
    QUIET_UNCERTAIN = True
    pend = sorted(memory.pending, key=lambda m: (m["court"], m["clean_name"], m["ucid"], m["raw_name"]))
    changed = True
    while changed:
        changed = False
        rest = []
        for m in pend:
            if m["context"].get("docket"):
                trim_to_known_judge(m)   # same trimming as in the matching pass, now that more judges may be known
            if is_first_name_only(m["clean_name"]):
                eid = same_case_first_name(m)
            else:
                r = match_judge(m, memory.entities)
                eid = r if r and r != UNRESOLVED else None
            if eid:
                attach_judge(m, eid, stats)
                changed = True
            else:
                rest.append(m)
        pend = rest
        groups = defaultdict(list)
        for m in pend:
            if len(person_tokens(m["clean_name"])[0]) >= 2 and m["context"].get("docket"):
                groups[(m["court"], m["clean_name"])].append(m)
        # promote ONE qualified name per round (most cases first), then re-match, so variants of a just-created judge
        # ("<name> III", "<first> <last>" vs "<first> <middle> <last>") join it instead of becoming separate judges
        qualified = []
        for key in sorted(groups):
            ms = groups[key]
            n_cases = len({x["ucid"] for x in ms if x["context"].get("docket_title")})
            core = person_tokens(key[1])[0]
            on_fjc = len(core) >= 2 and (core[0], core[-1]) in FJC_NAMES   # option A: FJC-listed name, title seen once is enough
            if n_cases >= DOCKET_CREATE_MIN_CASES or (on_fjc and n_cases >= 1):
                qualified.append((-n_cases, key))
        if qualified:
            key = sorted(qualified)[0][1]
            ms = groups[key]
            eid = create_judge(ms[0], stats)
            for x in ms[1:]:
                attach_judge(x, eid, stats)
            done_ids = {id(x) for x in ms}
            pend = [m for m in pend if id(m) not in done_ids]
            changed = True
    memory.pending = pend
    QUIET_UNCERTAIN = False
    stats["pending_judge_mentions_left"] = len(pend)


DROPPED = defaultdict(int)  # drop counts by reason, reported in the run summary


def drop_row(rec, reason, stage, field=None):
    DROPPED[reason.split(":")[0] if stage != "stage2" else reason.split(" (")[0]] += 1
    return {"entity_type": rec.get("entity_type"), "raw_name": rec.get("raw_name"), "clean_name": rec.get("clean_name"),
            "ucid": rec.get("ucid"), "court": rec.get("court"), "field": field or FIELD_OF.get(rec.get("entity_type"), ""),
            "reason": reason, "stage": stage}


def type_gate_stage1(rec):
    """('keep' | 'drop' | 'undecided', reason) for counsel / firm / party records, from record signals and v4_rules.yaml."""
    t, raw, ctx = rec["entity_type"], rec["raw_name"], rec["_ctx"]
    clean = counsel_core(rec["clean_name"]) if t == "counsel" else rec["clean_name"]
    if not JUNK_MUST_CONTAIN.search(rec["clean_name"]):
        return "drop", "junk: no letters"
    if clean in PLACEHOLDERS.get(t, ()) or (t == "firm" and clean.startswith(FIRM_PLACEHOLDER_PREFIXES)):
        return "drop", f"placeholder: '{clean}'"
    hit = STAFF_RE.search(clean)
    if hit:
        if t in ("counsel", "firm"):
            return "drop", f"court staff: '{hit.group(0)}' in the name"
        rest = [w for w in STAFF_RE.sub(" ", clean).split()]
        if len(rest) <= PARTY_STAFF_MAX_OTHER and not is_org_name(raw, rec["clean_name"]):
            return "drop", f"court staff: '{hit.group(0)}' party name"
    if t in ("counsel", "firm") and federal_agency(clean):
        return "keep", "federal office name (decided by Stage 1, never sent to Stage 2)"
    if t == "counsel":
        if ctx.get("rawinfo_staff") and not ctx.get("emails"):
            return "drop", f"court staff: '{ctx['rawinfo_staff'][0]}' in raw_info, no email"
        if counsel_has_contact(ctx) or ctx.get("bar"):
            return "keep", "lawyer evidence (contact / firm / bar)"
        if len(INDEX.counsel_name_cases.get(clean, ())) >= UNDECIDED_MIN_CASES:
            return "undecided", f"no contact info, name in {UNDECIDED_MIN_CASES}+ cases"
        return "keep", "no contact info, few cases"
    return "keep", "default"


def type_cache_key(rec):
    t = rec["entity_type"]
    clean = counsel_core(rec["clean_name"]) if t == "counsel" else rec["clean_name"]
    return f"{t}|{clean}|no-contact-many-cases"


_GLINER = None


def classify_with_model(keys_raw):
    """Stage 2: GLiNER2 zero-shot labels for {cache_key: raw_name}; answers go to memory.type_cache."""
    global _GLINER
    if not keys_raw:
        return 0
    if _GLINER is None:
        import warnings
        warnings.filterwarnings("ignore")
        from gliner2 import GLiNER2
        print(f"Loading {TYPE_MODEL_CFG['name']} ...", flush=True)
        _GLINER = GLiNER2.from_pretrained(TYPE_MODEL_CFG["name"])
    labels = TYPE_MODEL_CFG["labels"]
    keys = sorted(keys_raw)
    res = _GLINER.batch_classify_text([keys_raw[k] for k in keys], {"type": list(labels)}, include_confidence=True)
    for k, r in zip(keys, res):
        lab = r["type"]["label"] if isinstance(r.get("type"), dict) else r.get("type")
        conf = float(r["type"].get("confidence", 0)) if isinstance(r.get("type"), dict) else 0.0
        memory.type_cache[k] = {"key": k, "name": keys_raw[k], "label": labels.get(lab, lab), "confidence": round(conf, 4),
                                "model": TYPE_MODEL_CFG["name"]}
    return len(keys)


def extract_mentions_from_json(json_content, filename):
    """Returns (records, drops). Every item that is not kept is returned in drops with a reason."""
    data = json.loads(json_content)
    ucid = data.get("ucid")
    if not ucid:
        return [], [{"entity_type": None, "raw_name": None, "clean_name": None, "ucid": None, "court": None,
                     "field": "ucid", "reason": f"file has no ucid: {filename}", "stage": "extraction"}]
    court = data.get("court", "unknown")
    extracted = []  # (raw_name, etype, context)
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
            ri_staff = STAFF_RE.search(" ".join(re.sub(r"<[^>]+>|[^\w\s]", " ", raw_info.lower()).split()))
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
                }))
                p_ctx["firm_raw"].append(office.strip())

    # 2. Extract Judges Mentioned Only in Docket Text
    if EXTRACTION_CONFIG["judge"]["docket_scan"]:
        for item in data.get("docket", []):
            txt = item.get("docket_text", "")
            for pattern in DOCKET_JUDGE_PATTERNS:
                for mt in pattern.finditer(txt):
                    candidate = mt.group(1)
                    cand_clean = candidate.strip().lower()
                    if not any(stopword in cand_clean for stopword in NON_JUDGE_DOCKET_KEYWORDS):
                        extracted.append((candidate.strip(), "judge", {"docket": [True],
                                          "docket_title": [True] if DOCKET_TITLE_RE.search(mt.group(0)) else []}))
                    else:
                        drops.append(drop_row(base_rec(candidate, "judge"), "docket: non-judge keyword", "extraction",
                                              "docket[].docket_text"))

    # Convert to unique records per file (same dedup as v4_run.py; contexts of repeats are merged)
    records = []
    seen = {}
    for raw_name, etype, ctx in extracted:
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
            if judge_is_non_name(clean):
                r_ = base_rec(raw_name, etype); r_["clean_name"] = clean
                drops.append(drop_row(r_, "judge: non-name word", "extraction",
                                      "docket[].docket_text" if ctx.get("docket") else None))
                continue
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
        for f, vals in ctx.items():
            for v in vals:
                if v and v not in rec["_ctx"][f]:
                    rec["_ctx"][f].append(v)
    party_names = sorted({r["clean_name"] for r in records if r["entity_type"] == "party"})
    for r in records:
        if r["entity_type"] == "party":
            r["_ctx"]["coparties"] = [p for p in party_names if p != r["clean_name"]]
            # organisation / acronym decision needs the court's acronym set, so it is made at match time
            # (party_org_decision); pass 1 only collects this name's initials for that set.
            r["_ctx"]["batch_initials"] = batch_initials(r["raw_name"], r["clean_name"])
    return records, drops


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
        recs, drops = extract_mentions_from_json(content.decode("utf-8", errors="replace"), fname)
        for r in recs:
            if r["entity_type"] == "counsel":
                INDEX.counsel_name_cases[counsel_core(r["clean_name"])].add(r["ucid"])
            if r["entity_type"] == "party":
                INDEX.add_party_name_case(r["clean_name"], r["ucid"])
                INDEX.add_org_initials(r["_ctx"], r["court"])
            elif r["entity_type"] == "judge":
                INDEX.add_judge_first(r["clean_name"], r["court"])
        todo.append((fname, checksum, recs, drops))

    # Type gate: Stage 1 for every new counsel / firm / party record; Stage 2 (model) once per undecided name, cached.
    undecided = {}
    for _, _, recs, _ in todo:
        for r in recs or []:
            if r["entity_type"] in ("counsel", "firm", "party"):
                r["_gate"] = type_gate_stage1(r)
                if r["_gate"][0] == "undecided":
                    k = type_cache_key(r)
                    if k not in memory.type_cache:
                        undecided[k] = r["raw_name"]
    stats["type_gate_undecided_names_new"] = len(undecided)
    if TYPE_MODEL == "gliner2" and undecided:
        t_m = time.perf_counter()
        stats["type_model_names_classified"] = classify_with_model(undecided)
        stats["type_model_sec"] = round(time.perf_counter() - t_m, 1)

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
            if gate:
                decision, why = gate
                if decision == "undecided":
                    row = memory.type_cache.get(type_cache_key(r))
                    if row and row["label"] in TYPE_MODEL_CFG["drop_labels"] and row["confidence"] >= TYPE_MODEL_CFG["min_confidence"]:
                        memory.dropped.append(drop_row(r, f"stage2 {row['model']}: {row['label']} ({row['confidence']:.2f}); {why}", "stage2"))
                        continue
                    r["_ctx"]["type_gate"] = [f"kept: {why}; " + (f"model said {row['label']} ({row['confidence']:.2f})" if row else "no model answer")]
                    stats["type_gate_undecided_kept"] = stats.get("type_gate_undecided_kept", 0) + 1
                elif decision == "drop":
                    memory.dropped.append(drop_row(r, why, "stage1"))
                    continue
            kept.append(r)
        raw_mentions = kept
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
                    c.update(party_org_decision(m_data["raw_name"], m_data["clean_name"], m_data["court"]))
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
                        "canonical_name": judge_display_name(m_data["raw_name"]) if etype == "judge" else m_data["raw_name"],
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

    resolve_pending(stats)
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
    global STORAGE_DIR, ENTITIES_FILE, MENTIONS_FILE, STATE_FILE, UNCERTAIN_FILE, DROPPED_FILE, TYPE_CACHE_FILE, PENDING_FILE
    global TYPE_MODEL, memory
    ap = argparse.ArgumentParser(description="V4 incremental entity resolution, context-aware matching (local)")
    ap.add_argument("--input", nargs="+", required=True, help=".json file(s), folder(s) of .json files, or .zip")
    ap.add_argument("--output", required=True, help="folder for entities.jsonl, mentions.jsonl, state.json, uncertain_pairs.jsonl")
    ap.add_argument("--rules", default=None, help="rules file (default: v4_rules.yaml next to this script)")
    ap.add_argument("--type-model", choices=["none", "gliner2"], default="none",
                    help="Stage 2 of the type gate for names Stage 1 cannot decide (default: none = keep them)")
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
    TYPE_CACHE_FILE = STORAGE_DIR / "type_cache.jsonl"
    PENDING_FILE = STORAGE_DIR / "pending_judges.jsonl"
    TYPE_MODEL = args.type_model

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

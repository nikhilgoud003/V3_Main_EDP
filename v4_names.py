"""Shared name handling for Layer A (v4_run_ctx.py) and Layer B (v4_consolidate.py).

- NameGate: "what is this name?" -> PERSON / ORGANISATION / NOT_ENTITY from two models
  (spaCy NER, run on the sentence when there is one, and probablepeople Person vs Corporation),
  run once per distinct string and cached in a jsonl file; disagreements are settled by the caller
  (known entities / FJC), then optionally by a local Ollama model (cached), else sent to review.
- org_parts(): one organisation normalisation for both layers ("&" / "and" / "+", commas, periods,
  line breaks treated the same; legal-form parts removed with probablepeople's own
  CorporationLegalType component; text in parentheses or after a final " - " goes to an office field).
- display_name(): the most complete variant, rebuilt from the parser's components.
No person, firm, court or case names and no keyword lists for deciding what is a name live here.
"""
import functools
import json
import os
import re
import unicodedata
import urllib.request
from pathlib import Path

import probablepeople as pp

PERSON, ORG, NOT = "PERSON", "ORGANISATION", "NOT_ENTITY"
# spaCy's own label set mapped to the three outcomes (the model's label names, not a list of words)
SPACY_LABELS = {"PERSON": PERSON, "ORG": ORG, "GPE": ORG, "NORP": ORG, "FAC": ORG}
SPACY_COVER = 0.6   # an entity must cover this share of the name to count as a vote (choice; see README)
# organisation-type word the parser tags at the END of a name ("Company", "Corp.") also removed when 2+ name words remain;
# off by default (the spec removes CorporationLegalType parts only). Set from v4_rules.yaml organisations.legal_form_rule.
TRAILING_ORG_TYPE_IS_LEGAL = False


def fold(s):
    s = unicodedata.normalize("NFKD", str(s or ""))
    return "".join(c for c in s if not unicodedata.combining(c))


def squash(s):
    return " ".join(str(s or "").split())


def atomic_write_lines(path, rows):
    """Write jsonl rows to a temp file, then rename over the target (a crash never leaves a half-written file)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def atomic_write_text(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------- probablepeople
@functools.lru_cache(maxsize=None)
def pp_parse(text):
    """[(token, label)] from probablepeople; on a repeated-label error, the parser's own token labels."""
    try:
        return tuple(pp.parse(text))
    except Exception:
        return ()


@functools.lru_cache(maxsize=None)
def pp_type(text):
    """PERSON / ORGANISATION from probablepeople. When the parser reports a repeated label (an unusual but often real
    name), its own token labels decide: mostly Corporation* -> ORGANISATION, else PERSON."""
    try:
        _, typ = pp.tag(text)
    except pp.RepeatedLabelError as e:
        labs = [lab for _, lab in getattr(e, "parsed_string", [])]
        if not labs:
            return NOT
        return ORG if sum(lab.startswith("Corporation") for lab in labs) * 2 > len(labs) else PERSON
    except Exception:
        return NOT
    return ORG if typ == "Corporation" else PERSON


# ---------------------------------------------------------------- organisation normalisation
_PARENS = re.compile(r"\(([^)]*)\)")
_FINAL_DASH = re.compile(r"(?:\s[-–—]+\s*|\s*[-–—]+\s)([^-–—]+)$")   # a dash with a space on at least one side, at the end
_INITIAL_DOTS = re.compile(r"\b([A-Za-z](?:\.[A-Za-z])+\.?)")        # U.S.A. / L.L.C. / P.C -> USA / LLC / PC
_SEPARATORS = re.compile(r"[,;\n\r\t]+")


def _to_core(text):
    t = fold(text).lower().replace("'", "").replace("’", "")
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    toks = [w for w in t.split() if w != "and"]
    if toks and toks[0] == "the" and len(toks) > 1:   # a leading article (the only word rule here; see README)
        toks = toks[1:]
    return " ".join(toks)


def _join_letters(core):
    """Runs of single letters are one token ("a p green" and "ap green" -> "ap green"), like dotted initialisms."""
    out, run = [], []
    for w in core.split():
        if len(w) == 1 and w.isalpha():
            run.append(w)
            continue
        if run:
            out.append("".join(run) if len(run) > 1 else run[0])
            run = []
        out.append(w)
    if run:
        out.append("".join(run) if len(run) > 1 else run[0])
    return " ".join(out)


@functools.lru_cache(maxsize=None)
def org_parts(raw):
    """{'core', 'office', 'legal'} for an organisation name."""
    s = fold(raw).replace("&", " and ").replace("+", " and ")
    offices = [o for o in _PARENS.findall(s) if o.strip()]
    s = _PARENS.sub(" ", s)
    s = _SEPARATORS.sub(", ", s)
    m = _FINAL_DASH.search(s)
    if m and s[:m.start()].strip():
        offices.append(m.group(1))
        s = s[:m.start()]
    s = _INITIAL_DOTS.sub(lambda mm: mm.group(1).replace(".", ""), s)
    s = squash(s.replace(".", " "))
    keep, legal = [], []
    parsed = [(t, l) for t, l in pp_parse(s.replace(",", " , ")) if t != ","]
    for tok, lab in parsed:
        if lab == "CorporationLegalType":
            legal.append(tok)
        elif lab in ("CorporationNameBranchType", "CorporationNameBranchIdentifier"):
            offices.append(tok)
        else:
            keep.append((tok, lab))
    orgtype = [t for t, l in keep if l == "CorporationNameOrganization"]
    if TRAILING_ORG_TYPE_IS_LEGAL and len(keep) >= 3 and keep[-1][1] == "CorporationNameOrganization":
        legal.insert(0, keep.pop()[0])
    keep = [t for t, _ in keep]
    if not parsed:
        keep = s.replace(",", " ").split()
    core = _join_letters(_to_core(" ".join(keep)) or _to_core(s))
    office = _to_core(" ".join(offices))
    # "loose" core: without the words the parser tags as organisation types (Company, Corp., International ...);
    # a Layer B feature only, never a Layer A key
    loose = " ".join(w for w in core.split() if w not in set(_to_core(" ".join(orgtype)).split())) or core
    return {"core": core, "office": office, "legal": _to_core(" ".join(legal)), "loose": loose}


def org_head(raw):
    """The raw name without its office parts (text in parentheses, text after a final " - ")."""
    s = _PARENS.sub(" ", fold(raw))
    s = _SEPARATORS.sub(", ", s)
    m = _FINAL_DASH.search(s)
    if m and s[:m.start()].strip():
        s = s[:m.start()]
    return squash(s).strip(" ,")


def org_key(raw):
    p = org_parts(raw)
    return p["core"] + (f" | {p['office']}" if p["office"] else "")


# ---------------------------------------------------------------- display names
PERSON_PARTS = (("GivenName", "FirstInitial"), ("Nickname",), ("MiddleName", "MiddleInitial"),
                ("Surname", "LastInitial"), ("SuffixGenerational",))
_PERSON_KEEP = {lab for grp in PERSON_PARTS for lab in grp if lab != "Nickname"}


def _case(word):
    return word.title() if word.isupper() and len(word) > 2 else word


def person_display(raw):
    """(n_components, text): a person name rebuilt from probablepeople components (no prefixes such as titles)."""
    s = squash(str(raw).replace("\n", " "))
    toks = [(t, l) for t, l in pp_parse(s) if l in _PERSON_KEEP]
    labs = [l for _, l in toks]
    # a leading token the parser calls a surname, with a given name and another surname after it, is a prefix ("Hon.")
    if len(toks) >= 3 and labs[0] == "Surname" and "GivenName" in labs[1:] and "Surname" in labs[labs.index("GivenName", 1):]:
        toks = toks[1:]
    if not toks:
        return 0, s.strip(" ,;.-")
    words = [t.strip(",") if l == "SuffixGenerational" else _case(t.strip(",")) for t, l in toks]
    return len({l for _, l in toks}), squash(" ".join(words)).strip(" ,;-")


def org_display(raw):
    s = squash(_SEPARATORS.sub(", ", str(raw))).strip(" ,;-")
    return len(org_parts(raw)["core"].split()), s


def display_name(raws, kind):
    """Most complete variant, rebuilt from the parser's components. kind: PERSON or ORGANISATION."""
    best = None
    for r in sorted(set(x for x in raws if x and str(x).strip())):
        n, text = person_display(r) if kind == PERSON else org_display(r)
        k = (n, len(text), text)
        if best is None or k > best[0]:
            best = (k, text)
    return best[1] if best else ""


# ---------------------------------------------------------------- the gate
def prep_for_ner(text):
    text = squash(text)
    return text.title() if text.isupper() else text


class NameGate:
    """Two votes per distinct string, cached in <cache_path> (jsonl); LLM answers cached in the same rows."""

    def __init__(self, cache_path, spacy_model="en_core_web_md", llm_cfg=None):
        self.cache_path = Path(cache_path) if cache_path else None
        self.spacy_model = spacy_model
        self.llm_cfg = llm_cfg
        self._nlp = None
        self.cache = {}
        self.new = 0
        if self.cache_path and self.cache_path.exists():
            for line in open(self.cache_path, encoding="utf-8"):
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["key"]] = row

    @staticmethod
    def key(name, sentence=None):
        return ("s|" if sentence else "n|") + squash(name)

    def nlp(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load(self.spacy_model, disable=["parser", "lemmatizer", "tagger", "attribute_ruler"])
        return self._nlp

    @staticmethod
    def _spacy_vote(doc, start, end):
        span = max(1, end - start)
        best = None
        for e in doc.ents:
            ov = min(end, e.end_char) - max(start, e.start_char)
            if ov >= SPACY_COVER * span and e.label_ in SPACY_LABELS:
                lab = SPACY_LABELS[e.label_]
                if best is None or (lab == PERSON and best != PERSON):
                    best = lab
        return best or NOT

    def vote(self, items, progress=None):
        """items: {key: (name, sentence or None, (start, end) or None)}. Fills the cache for keys not yet seen."""
        todo = [k for k in sorted(items) if k not in self.cache]
        if not todo:
            return 0
        texts, spans = [], []
        for k in todo:
            name, sent, span = items[k]
            if sent and span:
                texts.append(sent)
                spans.append(span)
            else:
                t = prep_for_ner(name)
                texts.append(t)
                spans.append((0, len(t)))
        nlp = self.nlp()
        for i, doc in enumerate(nlp.pipe(texts, batch_size=512)):
            k = todo[i]
            name = items[k][0]
            self.cache[k] = {"key": k, "name": squash(name), "spacy": self._spacy_vote(doc, *spans[i]),
                             "pp": pp_type(squash(name))}
            if progress and (i + 1) % 20000 == 0:
                progress(f"name gate: {i + 1}/{len(todo)} new strings")
        self.new += len(todo)
        return len(todo)

    def votes_of(self, name, sentence=None):
        row = self.cache.get(self.key(name, sentence))
        return (row["spacy"], row["pp"]) if row else (None, None)

    def llm(self, name, role, sentence=None):
        """Ollama answer (cached): PERSON / ORGANISATION / NOT_ENTITY, or None when the LLM is off or fails."""
        if not self.llm_cfg:
            return None
        k = self.key(name, sentence)
        row = self.cache.setdefault(k, {"key": k, "name": squash(name)})
        if "llm" in row:
            return row["llm"]
        prompt = (f'In a US federal court record, the text "{squash(name)}" appears as a {role}.'
                  + (f' Sentence: "{squash(sentence)[:400]}".' if sentence else "")
                  + ' Is it the name of a PERSON, an ORGANISATION, or NOT AN ENTITY (a role, an event or other text)? '
                  'Reply with JSON only: {"label": "PERSON" | "ORGANISATION" | "NOT_ENTITY"}')
        body = {"model": self.llm_cfg["model"], "prompt": prompt, "stream": False, "format": "json",
                "options": {"temperature": 0, "seed": 0}}
        try:
            req = urllib.request.Request(self.llm_cfg["url"] + "/api/generate", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            text = json.loads(urllib.request.urlopen(req, timeout=self.llm_cfg.get("timeout_s", 60)).read())["response"]
            lab = str(json.loads(text).get("label", "")).upper().replace(" ", "_")
            lab = {"ORGANIZATION": ORG, "NOT_AN_ENTITY": NOT}.get(lab, lab)
            row["llm"] = lab if lab in (PERSON, ORG, NOT) else None
        except Exception:
            row["llm"] = None
        self.new += 1
        return row["llm"]

    def save(self):
        if self.cache_path:
            atomic_write_lines(self.cache_path, (self.cache[k] for k in sorted(self.cache)))


def ollama_available(url, model):
    try:
        tags = json.loads(urllib.request.urlopen(url + "/api/tags", timeout=5).read())
        return model in {m["name"] for m in tags.get("models", [])}
    except Exception:
        return False

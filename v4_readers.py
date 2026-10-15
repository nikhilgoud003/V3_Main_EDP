"""Input readers for Layer A, chosen by the file's STRUCTURE (never its name):

  docket JSON    a dict with "ucid" and "parties" / "docket"        -> v4_run_ctx.extract_mentions_from_json
  OCR document   a dict with "pages": [{"blocks": [{"text", ...}]}]  -> read_text_files()
  anything else  any JSON                                            -> read_text_files() (every text field + its JSON path)

Document and generic readers find judge / party / counsel / firm names with a model (GLiNER2 with label
descriptions, or a local Ollama model returning JSON), run once per distinct text block and cached. Case number
and court come from the text and are cross-checked with the file name. Page, bbox and confidence are kept;
low-confidence names go to a review file instead of becoming entities.
"""
import hashlib
import json
import re
import urllib.request
from pathlib import Path

# structural patterns (formats, not names)
CASE_RE = re.compile(r"\b(\d{1,2}):(\d{2})-([a-z]{2,5})-(\d{3,6})\b", re.I)
FILE_RE = re.compile(r"^([a-z]{2,6})(?:;;|_)(\d{1,2})-(\d{2})-([a-z]{2,5})-(\d{3,6})", re.I)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\(?\b\d{3}\)?[-.\s]\d{3}[-.]\d{4}\b")   # needs separators (dates and case numbers are not phones)


def file_kind(data):
    if isinstance(data, dict) and data.get("ucid") and ("parties" in data or "docket" in data):
        return "docket"
    if isinstance(data, dict) and isinstance(data.get("pages"), list) and any(
            isinstance(p, dict) and isinstance(p.get("blocks"), list) for p in data["pages"]):
        return "document"
    return "generic"


def case_from_filename(filename):
    m = FILE_RE.match(Path(filename).name)
    if not m:
        return None, None
    court, div, yy, typ, num = m.groups()
    return court.lower(), f"{div}:{yy}-{typ.lower()}-{num}"


def case_from_text(texts):
    for t in texts:
        m = CASE_RE.search(t or "")
        if m:
            div, yy, typ, num = m.groups()
            return f"{int(div)}:{yy}-{typ.lower()}-{num}"
    return None


class Extractor:
    """Name extraction from free text, one model call per distinct text, cached in <cache_path>."""

    def __init__(self, cfg, cache_path, mode="gliner2", llm_cfg=None):
        self.cfg, self.mode, self.llm_cfg = cfg, mode, llm_cfg
        self.cache_path = Path(cache_path)
        self.cache = {}
        self._model = None
        self.new = 0
        if self.cache_path.exists():
            for line in open(self.cache_path, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    self.cache[r["key"]] = r

    def _key(self, text, labels=None):
        labels = json.dumps(labels or self.cfg["labels"], sort_keys=True)   # a new label set asks the model again
        return hashlib.sha256(f"{self.mode}|{labels}|{text}".encode()).hexdigest()

    def model(self):
        if self._model is None:
            import warnings
            warnings.filterwarnings("ignore")
            from gliner2 import GLiNER2
            self._model = GLiNER2.from_pretrained(self.cfg["gliner2_model"])
        return self._model

    def run(self, texts, labels=None):
        """{text: [ {label, text, confidence, start, end} ]} for every text (cached by text + label set)."""
        labels = dict(labels or self.cfg["labels"])
        key = lambda t: self._key(t, labels)
        todo = sorted({t for t in texts if t and key(t) not in self.cache})
        if todo and self.mode == "gliner2":
            res = self.model().batch_extract_entities(todo, labels, batch_size=8, threshold=self.cfg["model_threshold"],
                                                      include_confidence=True, include_spans=True)
            for t, r in zip(todo, res):
                ents = [{"label": lab, "text": e["text"], "confidence": round(float(e.get("confidence", 0)), 4),
                         "start": e.get("start"), "end": e.get("end")}
                        for lab, lst in (r.get("entities") or {}).items() for e in lst]
                self.cache[key(t)] = {"key": key(t), "entities": ents}
        elif todo and self.mode == "ollama":
            for t in todo:
                self.cache[key(t)] = {"key": key(t), "entities": self._ollama(t, labels)}
        self.new += len(todo)
        return {t: self.cache[key(t)]["entities"] for t in texts if t and key(t) in self.cache}

    def spacy_ents(self, texts, nlp_fn):
        """{text: [(label, start, end)]} from spaCy NER (cascade step 1), cached by block text."""
        kfun = lambda t: "spacy|" + hashlib.sha256(t.encode()).hexdigest()
        todo = sorted({t for t in texts if t and kfun(t) not in self.cache})
        if todo:
            nlp = nlp_fn()
            prep = [t.title() if sum(c.isupper() for c in t) > 0.6 * max(1, sum(c.isalpha() for c in t)) else t for t in todo]
            for t, doc in zip(todo, nlp.pipe(prep, batch_size=64)):
                self.cache[kfun(t)] = {"key": kfun(t), "ents": [(e.label_, e.start_char, e.end_char) for e in doc.ents]}
        return {t: self.cache[kfun(t)]["ents"] for t in texts if t and kfun(t) in self.cache}

    def _ollama(self, text, labels):
        prompt = ("Extract the people and organisations named in this US court document text. Use only these types: "
                  + json.dumps(labels) + '. Reply with JSON only: {"entities": [{"name": "...", "type": "<one of the types>", '
                  '"role": "...", "side": "plaintiff | defendant | other", "email": "", "phone": "", "address": "", '
                  '"firm": "", "court": "", "case_number": ""}]}\nText:\n' + text[:4000])
        body = {"model": self.llm_cfg["model"], "prompt": prompt, "stream": False, "format": "json",
                "options": {"temperature": 0, "seed": 0}}
        try:
            req = urllib.request.Request(self.llm_cfg["url"] + "/api/generate", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            out = json.loads(json.loads(urllib.request.urlopen(req, timeout=self.llm_cfg.get("timeout_s", 120)).read())["response"])
        except Exception:
            return []
        ents = []
        for e in out.get("entities", []) if isinstance(out, dict) else []:
            lab = str(e.get("type", "")).lower()
            if lab in labels and e.get("name"):
                start = text.find(e["name"])
                ents.append({"label": lab, "text": e["name"], "confidence": None, "start": start if start >= 0 else None,
                             "end": start + len(e["name"]) if start >= 0 else None})
            for extra in ("email", "phone"):
                if e.get(extra):
                    ents.append({"label": extra, "text": e[extra], "confidence": None, "start": None, "end": None})
        return ents

    def classify(self, texts, task, labels):
        """{text: (key, confidence)} - zero-shot class of each text (cached). labels: {description: key}."""
        names = list(labels)
        kfun = lambda t: hashlib.sha256(f"{self.mode}|classify|{task}|{'|'.join(names)}|{t}".encode()).hexdigest()
        todo = sorted({t for t in texts if t and kfun(t) not in self.cache})
        if todo and self.mode == "gliner2":
            res = self.model().batch_classify_text(todo, {task: names}, include_confidence=True)
            for t, r in zip(todo, res):
                v = r.get(task)
                lab = v.get("label") if isinstance(v, dict) else v
                conf = float(v.get("confidence", 0)) if isinstance(v, dict) else 0.0
                self.cache[kfun(t)] = {"key": kfun(t), "class": labels.get(lab, "other"), "confidence": round(conf, 4)}
        elif todo and self.mode == "ollama":
            for t in todo:
                prompt = ("Classify this text from a US court document. Choose exactly one of: " + json.dumps(names)
                          + '. Reply with JSON only: {"label": "<one of them>"}\nText:\n' + t[:2000])
                body = {"model": self.llm_cfg["model"], "prompt": prompt, "stream": False, "format": "json",
                        "options": {"temperature": 0, "seed": 0}}
                try:
                    req = urllib.request.Request(self.llm_cfg["url"] + "/api/generate", json.dumps(body).encode(),
                                                 {"Content-Type": "application/json"})
                    lab = json.loads(json.loads(urllib.request.urlopen(req, timeout=self.llm_cfg.get("timeout_s", 120)).read())["response"]).get("label")
                except Exception:
                    lab = None
                self.cache[kfun(t)] = {"key": kfun(t), "class": labels.get(lab, "other"), "confidence": None}
        self.new += len(todo)
        return {t: (self.cache[kfun(t)]["class"], self.cache[kfun(t)]["confidence"]) for t in texts if t and kfun(t) in self.cache}

    def remember_lines(self, filename, lines, ucid):
        """Normalised lines of a document (and its case), kept so later runs learn templates from every document seen."""
        self.cache[f"lines|{filename}"] = {"key": f"lines|{filename}", "lines": sorted(set(lines)), "ucid": ucid}

    def known_lines(self):
        """{filename: (set of normalised lines, ucid)} for every document read so far."""
        return {k[6:]: (set(v["lines"]), v.get("ucid")) for k, v in self.cache.items() if k.startswith("lines|")}

    def save(self):
        tmp = self.cache_path.with_name(self.cache_path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for k in sorted(self.cache):
                f.write(json.dumps(self.cache[k]) + "\n")
        tmp.replace(self.cache_path)


def norm_line(text):
    """Line text for template statistics: lower case, digits as '#', punctuation as spaces."""
    t = re.sub(r"\d", "#", (text or "").lower())
    return " ".join(re.sub(r"[^a-z# ]", " ", t).split())


def doc_blocks(data, kind):
    """[{'id', 'page', 'bbox', 'conf', 'class', 'text', 'lines', 'path'}] - OCR blocks, or one block per JSON text field."""
    out = []
    if kind == "document":
        for p in data.get("pages") or []:
            for k, b in enumerate((p.get("blocks") or []) if isinstance(p, dict) else []):
                t = (b.get("text") or "").strip() if isinstance(b, dict) else ""
                if t:
                    out.append({"id": f"p{p.get('page')}b{k}", "page": p.get("page"), "bbox": b.get("bbox"),
                                "conf": b.get("confidence"), "class": b.get("class"), "text": t, "lines": t.split("\n"),
                                "path": None})
    else:
        def walk(x, path):
            if isinstance(x, dict):
                for k in sorted(x, key=str):
                    walk(x[k], f"{path}.{k}")
            elif isinstance(x, list):
                for i, v in enumerate(x):
                    walk(v, f"{path}[{i}]")
            elif isinstance(x, str) and re.search(r"[A-Za-z]{2}", x):
                out.append({"id": path, "page": None, "bbox": None, "conf": None, "class": None, "text": x.strip(),
                            "lines": x.strip().split("\n"), "path": path})
        walk(data, "$")
    return out


def line_y(block, i):
    """Vertical centre of line i of a block (lines spread evenly over the block's bbox)."""
    bb = block["bbox"]
    if not bb or len(bb) < 4:
        return None
    n = max(1, len(block["lines"]))
    return bb[1] + (i + 0.5) * (bb[3] - bb[1]) / n


def line_score(a, b, **kw):
    """Similarity of two normalised lines (rapidfuzz ratio)."""
    from rapidfuzz import fuzz
    la, lb = len(a), len(b)
    if not la or not lb:
        return 0.0
    return fuzz.ratio(a, b)


def template_lines(doc_lines, sim):
    """{normalised line: (documents, cases)} with a similar line (fuzzy, for OCR noise), over every document read."""
    from rapidfuzz import fuzz, process
    by_line = {}
    for f, (lines, ucid) in doc_lines.items():
        for ln in lines:
            if len(ln) >= 3 and re.search("[a-z]", ln):
                by_line.setdefault(ln, set()).add((f, ucid))
    uniq = sorted(by_line)
    out = {}
    for ln in uniq:
        hits = set()
        for other, score, _ in process.extract(ln, uniq, scorer=line_score, score_cutoff=sim * 100, limit=None):
            hits |= by_line[other]
        out[ln] = (len({f for f, _ in hits}), len({u for _, u in hits}))
    return out


def ngram_stats(doc_lines, n=3):
    """{word n-gram: set of (document, case)} over every document read."""
    grams = {}
    for f, (lines, ucid) in doc_lines.items():
        for ln in lines:
            w = ln.split()
            for i in range(len(w) - n + 1):
                grams.setdefault(tuple(w[i:i + n]), set()).add((f, ucid))
    return grams


def ngram_template(line, grams, min_docs, n=3):
    """(share of the line's words inside repeated n-grams, cases those n-grams come from)."""
    w = line.split()
    if len(w) < n:
        return 0.0, 0
    cov, cases = [False] * len(w), set()
    for i in range(len(w) - n + 1):
        hits = grams.get(tuple(w[i:i + n]), set())
        if len({f for f, _ in hits}) >= min_docs:
            cases |= {u for _, u in hits}
            for k in range(i, i + n):
                cov[k] = True
    return sum(cov) / len(w), len(cases)


def labels_of(cfg_part):
    return {k: v for k, v in cfg_part.items() if k != "source"}


def join_blocks(blocks, gap_px, prose_words):
    """Reading units: consecutive short (non-prose) blocks of a page that overlap horizontally and nearly touch
    vertically are one unit, so a name split by the layout model (a caption party with a long d/b/a) is read whole."""
    out = []
    for b in blocks:
        words = lambda x: [ln for ln in x["lines"] if ln.strip()]
        short = lambda x: bool(words(x)) and sum(len(re.findall(r"[A-Za-z]{2,}", ln)) for ln in words(x)) / len(words(x)) < prose_words
        prev = out[-1] if out else None
        if (prev and prev["page"] == b["page"] and prev["bbox"] and b["bbox"] and short(prev) and short(b)
                and 0 <= b["bbox"][1] - prev["bbox"][3] <= gap_px
                and min(prev["bbox"][2], b["bbox"][2]) - max(prev["bbox"][0], b["bbox"][0])
                >= 0.5 * min(prev["bbox"][2] - prev["bbox"][0], b["bbox"][2] - b["bbox"][0])):
            prev["lines"] = prev["lines"] + b["lines"]
            prev["text"] = "\n".join(prev["lines"])
            prev["bbox"] = [min(prev["bbox"][0], b["bbox"][0]), prev["bbox"][1], max(prev["bbox"][2], b["bbox"][2]), b["bbox"][3]]
            prev["id"] += "+" + b["id"]
            continue
        out.append(dict(b))
    return out


def legal_part(name):
    """probablepeople finds a legal-form component (Inc., LLC, Pty Ltd ...) in the name: positive organisation evidence."""
    from v4_names import pp_parse
    return any(lab == "CorporationLegalType" for _, lab in pp_parse(" ".join(name.split())))


ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")
CITY_ZIP_RE = re.compile(r"[A-Za-z][A-Za-z .'-]+,\s*[A-Z][A-Za-z.]{1,13}\.?\s+\d{5}(?:-\d{4})?\b")   # "City, ST 12345" address form


def contact_evidence(b, cfg):
    """Contact details in a unit that structure can confirm: an email, a phone with separators, an address entity with a
    ZIP code, a bar-number entity that is not a case number."""
    out = set()
    if EMAIL_RE.search(b["text"]):
        out.add("email")
    if PHONE_RE.search(b["text"]):
        out.add("phone")
    for e in b["ents"]:
        if (e.get("confidence") or 0) < cfg["contact_min_confidence"]:
            continue
        if e["label"] == "address" and CITY_ZIP_RE.search(e["text"]):
            out.add("address")
        if e["label"] == "bar number" and re.search(r"\d", e["text"]) and not CASE_RE.search(e["text"]) \
                and not re.search(r"\d{1,2}:\d{2}", e["text"]):
            out.add("bar number")
    if any(CITY_ZIP_RE.search(ln) for ln in b["lines"]):
        out.add("address")
    return sorted(out)


def calibrate_relative(scores_ok, default_t, grid, tolerance, min_support):
    """Lowest threshold t (<= default_t) such that names scoring >= t pass the gate as often as names scoring >= default_t,
    within one standard error of that share (binomial, sqrt(p(1-p)/n)) or `tolerance`, whichever is larger.
    Returns (t, share at t, share at default_t, support at t, allowed drop)."""
    def share(t):
        sel = [ok for sc, ok in scores_ok if sc >= t]
        return (sum(sel) / len(sel) if sel else None), len(sel)
    base, n_base = share(default_t)
    if base is None or n_base < min_support:
        return default_t, base, base, n_base, None
    drop = max(tolerance, (base * (1 - base) / n_base) ** 0.5)
    best = (default_t, round(base, 3), round(base, 3), n_base, round(drop, 3))
    for t in sorted(grid, reverse=True):
        if t >= default_t:
            continue
        sh, n = share(t)
        if sh is not None and sh >= base - drop:
            best = (t, round(sh, 3), round(base, 3), n, round(drop, 3))
        else:
            break
    return best


def calibrate(scores_ok, grid, target, min_support):
    """Lowest threshold t in grid such that, among items with score >= t, at least `target` are ok (and min_support
    items). scores_ok: [(score, ok)]. Returns (t or None, share, support)."""
    best = None
    for t in sorted(grid, reverse=True):
        sel = [ok for sc, ok in scores_ok if sc >= t]
        if len(sel) >= min_support and sum(sel) / len(sel) >= target:
            best = (t, round(sum(sel) / len(sel), 3), len(sel))
        elif len(sel) >= min_support:
            break
    return best or (None, None, 0)


def read_text_files(items, extractor, cfg, gate, docket_roles, debug=None):
    """All document / generic files of a run at once.
    items: [(filename, data, kind)]; gate: v4_names.NameGate (spaCy + probablepeople votes, cached);
    docket_roles: {ucid: {name key: entity type}} from structured docket data of the same case.
    Returns ({filename: (ucid, court, extracted, reviews)}, mentioned_only rows, report rows).

    cascade: spaCy NER on every reading unit (cached) -> GLiNER2 only on units with a spaCy person / organisation
    or contact details (cached by unit text). Then:
    1. template lines: repeated (fuzzily, or by word 3-grams) in 2+ documents and NOT a gate-approved name line;
       short template labels are paired with the nearest name line by layout; GLiNER2 reads each label's meaning
       (party / lawyer / judge / non-participant role / other)
    2. document gate: a person needs spaCy PERSON + probablepeople Person; an organisation needs spaCy ORG or an
       organisation-label score above a threshold calibrated on names both models agree on; anything else -> review
    3. participant vs mentioned by layout; a non-participant label (payee, victim, ...) makes the name mentioned-only
    4. relations: a signature block is linked to the firm / address block that follows it (also across a page
       break); a person with a bar number, firm or address attached is counsel; an organisation is a firm only when
       attached to a lawyer
    5. one role per name per case; docket roles of the same case override"""
    import time as _time
    tc, W = cfg["template"], cfg["one_role"]
    type_of = cfg["label_types"]
    min_conf = cfg["min_entity_confidence"]
    org_labels = set(cfg["organisation_labels"])
    name_labels = set(type_of) | {"mentioned"} | org_labels
    t_all = _time.perf_counter()
    docs, report = {}, {}
    for filename, data, kind in items:
        t0 = _time.perf_counter()
        blocks = join_blocks(doc_blocks(data, kind), tc["join_gap_px"], tc["prose_block_words"])
        f_court, f_case = case_from_filename(filename)
        t_case = case_from_text([b["text"] for b in blocks])
        reviews = []
        if f_case and t_case and f_case != t_case:
            reviews.append({"file": filename, "reason": f"case number in the text ({t_case}) differs from the file name ({f_case})"})
        case = f_case or t_case
        if not case:
            reviews.append({"file": filename, "reason": "no case number in the file name or the text"})
            case = "unknown-" + hashlib.sha256(filename.encode()).hexdigest()[:10]
        court = f_court or "unknown"
        ucid = f"{court};;{case}"
        for k, b in enumerate(blocks):
            b["order"] = k
        docs[filename] = {"kind": kind, "blocks": blocks, "ucid": ucid, "court": court, "reviews": reviews}
        extractor.remember_lines(filename, [norm_line(ln) for b in blocks for ln in b["lines"]], ucid)
        # cascade, step 1 (spaCy) and step 2 (GLiNER2 on candidate units only), timed per document
        sp = extractor.spacy_ents([b["text"] for b in blocks], gate.nlp)
        units_for_model = [b["text"] for b in blocks
                if any(lab in ("PERSON", "ORG", "GPE", "NORP", "FAC") for lab, _, _ in sp.get(b["text"], []))
                or EMAIL_RE.search(b["text"]) or PHONE_RE.search(b["text"])]
        found = extractor.run(units_for_model)
        for b in blocks:
            b["ents"] = found.get(b["text"], [])
            b["candidate"] = b["text"] in found
        report[filename] = {"file": filename, "ucid": ucid, "blocks": len(blocks), "blocks_sent_to_gliner2": len(set(units_for_model)),
                            "model_sec": round(_time.perf_counter() - t0, 2)}
    # the names the model found, and the document gate for each distinct name
    for d in docs.values():
        for b in d["blocks"]:
            starts, pos = [], 0
            for ln in b["lines"]:
                starts.append(pos)
                pos += len(ln) + 1
            b["starts"] = starts
            b["spans"] = []
            for e in b["ents"]:
                if e["label"] not in name_labels or e.get("start") is None or not e["text"].strip():
                    continue
                parts = [x for x in e["text"].split("\n") if x.strip()]
                while len(parts) > 1 and not re.search(r"[A-Za-z]{3}", parts[-1]):
                    parts.pop()          # a continuation line of 1-2 letter tokens is OCR debris, not part of the name
                text = " ".join(" ".join(parts).split()).strip(" ,;:.")
                st = e["start"] + (len(e["text"]) - len(e["text"].lstrip()))
                b["spans"].append({"name": text, "label": e["label"], "conf": e.get("confidence") or 0.0, "start": st,
                                   "line": max([i for i, s0 in enumerate(starts) if s0 <= st] or [0])})
    names = sorted({sp_["name"] for d in docs.values() for b in d["blocks"] for sp_ in b["spans"]})
    gate.vote({gate.key(n): (n, None, None) for n in names})
    org_score = {}
    for d in docs.values():
        for b in d["blocks"]:
            for sp_ in b["spans"]:
                if sp_["label"] in org_labels:
                    org_score[sp_["name"]] = max(org_score.get(sp_["name"], 0), sp_["conf"])
    # organisation threshold, calibrated: names both models call organisations vs names both call persons
    pseudo = []
    for n in names:
        s_, p_ = gate.votes_of(n)
        if s_ == p_ and s_ in ("PERSON", "ORGANISATION"):
            pseudo.append((org_score.get(n, 0.0), s_ == "ORGANISATION"))
    org_min, org_share, org_n = calibrate(pseudo, [x / 100 for x in range(int(cfg["gate"]["org_score_floor"] * 100), 100, 2)],
                                          cfg["gate"]["org_precision"], cfg["gate"]["org_min_support"])
    org_min = org_min if org_min is not None else 1.01

    def doc_gate(n):
        s_, p_ = gate.votes_of(n)
        if s_ == "PERSON" and p_ == "PERSON":
            return "PERSON", "spaCy PERSON + probablepeople Person"
        if s_ == "ORGANISATION":
            return "ORGANISATION", "spaCy ORG"
        if p_ == "ORGANISATION" and legal_part(n):
            return "ORGANISATION", "probablepeople Corporation with a legal-form part"
        if org_score.get(n, 0) >= org_min and p_ != "PERSON":
            return "ORGANISATION", f"organisation label score {org_score[n]:.2f} >= {org_min:.2f}"
        return None, f"gate failed (spaCy {s_}, probablepeople {p_}, organisation score {org_score.get(n, 0):.2f})"
    verdict = {n: doc_gate(n) for n in names}
    # template lines: repeated, label-like, and not a gate-approved name line
    known = extractor.known_lines()
    counts = template_lines(known, tc["line_similarity"])
    grams = ngram_stats(known)
    for d in docs.values():
        for b in d["blocks"]:
            b["tmpl"] = []
            for i, ln in enumerate(b["lines"]):
                n_docs, _ = counts.get(norm_line(ln), (0, 0))
                share, _ = ngram_template(norm_line(ln), grams, tc["min_documents"])
                letters = max(1, sum(c.isalpha() for c in ln))
                name_cov = sum(sum(c.isalpha() for c in sp_["name"]) for sp_ in b["spans"]
                               if sp_["line"] == i and verdict[sp_["name"]][0])
                repeated = n_docs >= tc["min_documents"] or share >= tc["ngram_share"]
                b["tmpl"].append(repeated and name_cov * 3 < letters)   # a third covered by approved names: a name line

    def has_name(b, j):
        return any(sp_["line"] == j and sp_["conf"] >= cfg["contact_min_confidence"] for sp_ in b["spans"])
    labels_seen = set()
    for d in docs.values():
        for b in d["blocks"]:
            b["fields"] = {}
        pairs = []      # (distance, label text, value block index, value line)
        for bi, b in enumerate(d["blocks"]):
            for i, is_t in enumerate(b["tmpl"]):
                ln = b["lines"][i]
                if not is_t or len(re.findall(r"[A-Za-z]{2,}", ln)) > tc["label_max_words"] or ln.rstrip().endswith("."):
                    continue
                cands = [(abs(j - i), 0 if j < i else 1, bi, j) for j in range(len(b["lines"]))
                         if j != i and not b["tmpl"][j] and has_name(b, j) and abs(j - i) <= tc["max_pair_lines"]]
                if not cands and line_y(b, i) is not None:
                    y = line_y(b, i)
                    for bj, o in enumerate(d["blocks"]):
                        if bj == bi or o["page"] != b["page"] or not o["bbox"]:
                            continue
                        if min(b["bbox"][2], o["bbox"][2]) < max(b["bbox"][0], o["bbox"][0]):
                            continue
                        for j in range(len(o["lines"])):
                            if o["tmpl"][j] or not has_name(o, j):
                                continue
                            gap = abs(line_y(o, j) - y)
                            if gap <= tc["max_pair_gap_px"]:
                                cands.append((1000 + gap / 1000.0, 0 if line_y(o, j) < y else 1, bj, j))
                if cands:
                    dist, _, bj, j = min(cands)
                    pairs.append((dist, ln.strip(), bj, j))
        # each value line keeps only its nearest label(s): a far label belongs to another value
        nearest = {}
        for dist, lab_text, bj, j in pairs:
            nearest[(bj, j)] = min(nearest.get((bj, j), dist), dist)
        for dist, lab_text, bj, j in pairs:
            if dist == nearest[(bj, j)]:
                d["blocks"][bj]["fields"].setdefault(j, []).append(lab_text)
                labels_seen.add(lab_text)
    # label meanings: the extraction model reads the label (best over its OCR variants); the classifier picks when the
    # label names several roles
    role_labels = labels_of(cfg["label_role_labels"])
    role_of_label = dict(cfg["label_role_types"])
    seen = sorted(labels_seen)
    field_cls = extractor.classify(seen, "field", labels_of(cfg["field_labels"]))
    label_ner = extractor.run(seen, role_labels)
    ner_roles = {}
    for lab_text in seen:
        span = {}
        for e in label_ner.get(lab_text, []):
            r_ = role_of_label.get(e["label"])
            if r_ and (e.get("confidence") or 0) >= tc["label_min_confidence"]:
                span[r_] = (max(span.get(r_, (0, 0))[0], len(e["text"])), max(span.get(r_, (0, 0))[1], e["confidence"]))
        ner_roles[lab_text] = span
    meaning = {}
    for lab_text in seen:
        group = [x for x in seen if line_score(norm_line(x), norm_line(lab_text)) >= tc["line_similarity"] * 100]
        span = {}
        for x in group:
            for r_, (ln_, cf) in ner_roles[x].items():
                old = span.get(r_, (0, 0))
                span[r_] = (max(old[0], ln_), max(old[1], cf))
        cls = field_cls.get(lab_text, ("other", None))
        if len(span) == 1:
            r_, (ln_, cf) = next(iter(span.items()))
            meaning[lab_text] = (r_, cf)
        elif len(span) > 1:
            top = sorted(span.items(), key=lambda kv: -kv[1][0])
            if cls[0] in span:
                meaning[lab_text] = (cls[0], cls[1])
            elif top[0][1][0] >= 1.5 * top[1][1][0]:
                meaning[lab_text] = (top[0][0], top[0][1][1])
            else:
                meaning[lab_text] = ("other", None)
        else:
            meaning[lab_text] = ("other", None)
        if meaning[lab_text][0] == "nonparticipant" and (meaning[lab_text][1] or 0) < tc["nonparticipant_min_confidence"]:
            meaning[lab_text] = ("other", None)   # a non-participant role needs a clear reading (payee, victim, ...)
    roles = ("party", "counsel", "judge")
    # relation scope: a unit, plus the firm / address unit that follows a signature (also across a page break)
    for d in docs.values():
        bl = d["blocks"]
        for b in bl:
            b["contact"] = contact_evidence(b, cfg)
            # law firms only: an organisation the model labels a law firm (a caption's other party is not a firm)
            b["orgs"] = sorted({sp_["name"] for sp_ in b["spans"] if verdict[sp_["name"]][0] == "ORGANISATION"
                                and sp_["label"] == "law firm" and sp_["conf"] >= min_conf})
            filled = [ln for ln in b["lines"] if ln.strip()]
            b["prose"] = bool(filled) and sum(len(re.findall(r"[A-Za-z]{2,}", ln)) for ln in filled) / len(filled) >= tc["prose_block_words"]
            b["people"] = sorted({sp_["name"] for sp_ in b["spans"] if verdict[sp_["name"]][0] == "PERSON"})
            b["linked"] = None
        for k, b in enumerate(bl):
            if not b["people"]:
                continue
            for o in bl[k + 1:]:
                if all(o["tmpl"]) or not any(ln.strip() for ln in o["lines"]):
                    continue           # page headers / footers (form text) between the two parts are skipped
                if not o["people"] and not o["prose"] and (o["orgs"] or o["contact"]):
                    b["linked"] = o       # the firm / address unit that follows a signature
                break
    # occurrences
    scores_paired = []
    occ = []
    for f, d in docs.items():
        for b in d["blocks"]:
            filled = [ln for ln in b["lines"] if ln.strip()]
            prose = bool(filled) and sum(len(re.findall(r"[A-Za-z]{2,}", ln)) for ln in filled) / len(filled) >= tc["prose_block_words"]
            spans = {}
            for sp_ in b["spans"]:
                if b["tmpl"][sp_["line"]] or not re.search(r"[A-Za-z]{3}", sp_["name"]):
                    continue
                k = (sp_["line"], norm_line(sp_["name"]).replace("#", "").strip())
                cur = spans.setdefault(k, {"name": sp_["name"], "labels": {}, "start": sp_["start"], "line": sp_["line"]})
                cur["labels"][sp_["label"]] = max(cur["labels"].get(sp_["label"], 0), sp_["conf"])
            for (line, key), sp_ in spans.items():
                fields = b["fields"].get(line, [])
                means = [meaning.get(x, ("other", None)) for x in fields]
                best = max(sp_["labels"].values())
                n_rep = counts.get(norm_line(b["lines"][line]), (0, 0))[0]
                # strong evidence: a participant form label points to the name, or the name fills its own line in
                # several documents (a frequent participant such as the government in criminal captions)
                strong = any(m[0] in roles for m in means) or (n_rep >= tc["min_documents"] and verdict[sp_["name"]][0] is not None)
                if strong:
                    scores_paired.append((best, verdict[sp_["name"]][0] is not None))
                ln = b["lines"][line]
                other_words = len(re.findall(r"[A-Za-z]{2,}", ln)) - len(re.findall(r"[A-Za-z]{2,}", sp_["name"]))
                occ.append({"file": f, "ucid": d["ucid"], "name": sp_["name"], "key": key, "labels": sp_["labels"],
                            "best": best, "fields": fields, "means": means, "strong": strong, "block": b,
                            "span": (sp_["start"], sp_["start"] + len(sp_["name"])), "line_text": ln.strip(),
                            "value_line": any(m[0] in roles for m in means) or (other_words <= tc["prose_min_other_words"] and not prose),
                            "where": {"page": b["page"], "bbox": b["bbox"], "block_confidence": b["conf"],
                                      "block_class": b["class"], "json_path": b["path"]}})
    # lower model-score threshold for names a strong participant label points to, calibrated on the data
    low_t, low_share, base_share, low_n, low_drop = calibrate_relative(
        scores_paired, min_conf, sorted({round(sc, 2) for sc, _ in scores_paired}), cfg["gate"]["paired_tolerance"],
        cfg["gate"]["paired_min_support"])
    gate_rows, kept = [], []
    for o in occ:
        need = low_t if o["strong"] else min_conf
        kind, why = verdict[o["name"]]
        row = {"file": o["file"], "name": o["name"], "decision": kind or "review", "why": why,
               "model_score": round(o["best"], 3), "needed_score": need}
        if o["best"] < need:
            row["decision"], row["why"] = "review", f"model score {o['best']:.2f} below {need:.2f}"
            o["block"]["_reviews"] = o["block"].get("_reviews", []) + [row]
        gate_rows.append(row)
        if row["decision"] == "review":
            docs[o["file"]]["reviews"].append({"file": o["file"], "ucid": o["ucid"], "name": o["name"], "labels": o["labels"],
                                               **{k: v for k, v in o["where"].items() if v is not None}, "reason": row["why"]})
            continue
        o["kind"] = kind
        kept.append(o)
    occ = kept
    # second model pass on value lines alone: judge / attorney evidence
    line_found = extractor.run(sorted({o["line_text"] for o in occ if o["value_line"]}))
    for o in occ:
        for e in line_found.get(o["line_text"], []) if o["value_line"] else []:
            if e["label"] in ("judge", "attorney") and norm_line(e["text"]).replace("#", "").strip() == o["key"]:
                o["labels"][e["label"]] = max(o["labels"].get(e["label"], 0), e.get("confidence") or 0)
    # votes
    counsel_scope, counsel_domains = set(), {}
    for o in occ:
        b = o["block"]
        scope = [b] + ([b["linked"]] if b["linked"] else [])
        o["scope"] = scope
        v = {}
        for lab, c in o["labels"].items():
            t = type_of.get(lab)
            if t:
                v[t] = v.get(t, 0) + c * (W["person_party_label_weight"] if (o["kind"] == "PERSON" and t == "party") else 1.0)
        for m in o["means"]:
            if m[0] in roles:
                v[m[0]] = v.get(m[0], 0) + W["label_weight"]
        o["nonparticipant"] = any(m[0] == "nonparticipant" for m in o["means"]) and not any(m[0] in roles for m in o["means"])
        if o["kind"] == "PERSON":
            v["counsel"] = v.get("counsel", 0) + v.pop("firm", 0)
            judge_field = any(m[0] == "judge" for m in o["means"])
            attached = (any(x["contact"] or x["orgs"] for x in scope) and len(b["people"]) <= W["contact_max_people"]
                        and not b["prose"])
            if attached and not judge_field and o["value_line"]:
                v["counsel"] = v.get("counsel", 0) + W["attached_counsel_weight"]   # bar number / firm / address attached
                counsel_scope.update(id(x) for x in scope)
                counsel_domains.setdefault(o["ucid"], set()).update(
                    m_.split("@", 1)[1].lower() for x in scope for m_ in EMAIL_RE.findall(x["text"]))
        o["votes"] = v
    for o in occ:
        if o["kind"] != "ORGANISATION":
            continue
        v = o["votes"]
        v.pop("judge", None)
        lawyer_side = v.pop("counsel", 0) + v.pop("firm", 0)
        doms = {m_.split("@", 1)[1].lower() for x in o["scope"] for m_ in EMAIL_RE.findall(x["text"])}
        if any(id(x) in counsel_scope for x in o["scope"]) or doms & counsel_domains.get(o["ucid"], set()):
            v["firm"] = max(lawyer_side, W["label_weight"])   # an organisation is a firm only when attached to a lawyer
    # one role per name per case
    agg = {}
    for o in occ:
        a_ = agg.setdefault((o["ucid"], o["key"]), {"votes": {}, "occ": [], "participant": False})
        a_["occ"].append(o)
        if o["value_line"] and not o["nonparticipant"]:
            a_["participant"] = True
            for t, w in o["votes"].items():
                a_["votes"][t] = a_["votes"].get(t, 0) + w
    if debug is not None:
        debug.extend({k: o[k] for k in ("file", "name", "labels", "fields", "means", "votes", "kind", "value_line", "line_text")} for o in occ)
        debug.append({"meanings": meaning, "org_min": org_min, "low_t": low_t, "scores_paired": scores_paired})
    out = {f: (d["ucid"], d["court"], [], d["reviews"]) for f, d in docs.items()}
    mentioned = []
    for (ucid, key), a_ in sorted(agg.items(), key=lambda kv: kv[0]):
        first = a_["occ"][0]
        dock = (docket_roles.get(ucid) or {}).get(key)
        vs = {t: w for t, w in a_["votes"].items() if t in ("party", "counsel", "judge", "firm") and w > 0}
        decided = None
        if dock:
            decided, why = dock, "docket JSON of the same case"
        elif a_["participant"] and vs:
            tot = sum(vs.values())
            top = max(sorted(vs), key=lambda t: vs[t])
            if vs[top] >= W["min_share"] * tot:
                decided, why = top, "form label + relations + model evidence over all documents of the case"
            else:
                out[first["file"]][3].append({"file": first["file"], "ucid": ucid, "name": first["name"],
                                              "votes": {t: round(w, 3) for t, w in sorted(vs.items())},
                                              "reason": "conflicting roles for this name in this case"})
                continue
        for o in a_["occ"]:
            if not decided or ((not o["value_line"] or o["nonparticipant"]) and not dock):
                mentioned.append({"file": o["file"], "ucid": ucid, "name": o["name"], "kind": o["kind"], "model_labels": o["labels"],
                                  **{k: v for k, v in o["where"].items() if v is not None},
                                  "reason": ("a non-participant form label points to it" if o["nonparticipant"]
                                             else "named only inside text" if not decided
                                             else f"named inside text (a {decided} of this case elsewhere)")})
                continue
            b = o["block"]
            bestlab = max(o["labels"], key=o["labels"].get)
            base = {"document": [True], "document_file": [o["file"]], "model_label": [bestlab], "confidence": [round(o["best"], 4)],
                    "field": o["fields"][:2], "role_decision": [why], "reader_kind": [o["kind"]],
                    "document_gate": [verdict[o["name"]][1]]}
            for k, v in o["where"].items():
                if v is not None:
                    base[k] = [",".join(map(str, v)) if isinstance(v, list) else v]
            if decided == "judge":
                ctx = {**base, "docket": [True], "docket_title": [True]}
            elif decided == "party":
                side = cfg["label_sides"].get(max((l for l in o["labels"] if l in cfg["label_sides"]), key=o["labels"].get, default="party"), "other")
                ctx = {**base, "counsel_raw": [], "firm_raw": [], "role_raw": [bestlab], "side": [side]}
            elif decided == "counsel":
                scope = o["scope"]
                firms = sorted({x["name"] for x in occ if x["block"] in scope and x["kind"] == "ORGANISATION" and x["votes"].get("firm")})
                one = len(b["people"]) <= 1
                emails = sorted({m_.lower() for x in scope for m_ in EMAIL_RE.findall(x["text"])})
                phones = sorted({re.sub(r"\D", "", m_) for x in scope for m_ in PHONE_RE.findall(x["text"])})
                ctx = {**base, "emails": emails if one else [], "phones": phones if one else [], "addresses": [], "offices": [],
                       "firm_raw": firms, "party_raw": [], "party_side": [], "bar": [True] if any("bar number" in x["contact"] for x in scope) else [],
                       "rawinfo_staff": []}
            else:
                scope = o["scope"]
                ctx = {**base, "addresses": [], "phones": sorted({re.sub(r"\D", "", m_) for x in scope for m_ in PHONE_RE.findall(x["text"])}),
                       "domains": sorted({m_.split("@", 1)[1].lower() for x in scope for m_ in EMAIL_RE.findall(x["text"])})}
            out[o["file"]][2].append((o["name"], decided, ctx, b["text"], o["span"]))
    # report rows: per document gate decisions and time, plus the calibrated thresholds
    rows = []
    shared = round(_time.perf_counter() - t_all - sum(r["model_sec"] for r in report.values()), 2)
    for f, r in report.items():
        g = [x for x in gate_rows if x["file"] == f]
        r["gate"] = {k: sum(1 for x in g if x["decision"] == k) for k in ("PERSON", "ORGANISATION", "review")}
        r["gate_failed_examples"] = sorted({x["name"] for x in g if x["decision"] == "review"})[:12]
        rows.append(r)
    rows.append({"calibration": {
        "organisation_label_threshold": None if org_min > 1 else org_min, "organisation_threshold_precision": org_share,
        "organisation_threshold_support": org_n, "organisation_pseudo_labels": len(pseudo),
        "paired_name_threshold": low_t, "paired_share_passing_gate_at_threshold": low_share,
        "paired_share_passing_gate_at_default": base_share, "paired_allowed_drop": low_drop, "paired_threshold_support": low_n,
        "default_threshold": min_conf}, "shared_sec": shared, "total_sec": round(_time.perf_counter() - t_all, 2)})
    return out, mentioned, rows

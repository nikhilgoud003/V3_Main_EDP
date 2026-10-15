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

    def _key(self, text):
        labels = json.dumps(self.cfg["labels"], sort_keys=True)   # a new label set asks the model again
        return hashlib.sha256(f"{self.mode}|{labels}|{text}".encode()).hexdigest()

    def model(self):
        if self._model is None:
            import warnings
            warnings.filterwarnings("ignore")
            from gliner2 import GLiNER2
            self._model = GLiNER2.from_pretrained(self.cfg["gliner2_model"])
        return self._model

    def run(self, texts):
        """{text: [ {label, text, confidence, start, end} ]} for every text (cached)."""
        todo = sorted({t for t in texts if t and self._key(t) not in self.cache})
        labels = dict(self.cfg["labels"])
        if todo and self.mode == "gliner2":
            res = self.model().batch_extract_entities(todo, labels, batch_size=8, threshold=self.cfg["model_threshold"],
                                                      include_confidence=True, include_spans=True)
            for t, r in zip(todo, res):
                ents = [{"label": lab, "text": e["text"], "confidence": round(float(e.get("confidence", 0)), 4),
                         "start": e.get("start"), "end": e.get("end")}
                        for lab, lst in (r.get("entities") or {}).items() for e in lst]
                self.cache[self._key(t)] = {"key": self._key(t), "entities": ents}
        elif todo and self.mode == "ollama":
            for t in todo:
                self.cache[self._key(t)] = {"key": self._key(t), "entities": self._ollama(t, labels)}
        self.new += len(todo)
        return {t: self.cache[self._key(t)]["entities"] for t in texts if t and self._key(t) in self.cache}

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


def read_text_files(items, extractor, cfg, kind_of, docket_roles, debug=None):
    """All document / generic files of a run at once.
    items: [(filename, data, kind)]; kind_of(name, model_label) -> 'PERSON' | 'ORGANISATION' | None;
    docket_roles: {ucid: {name key: entity type}} from structured docket data of the same case.
    Returns ({filename: (ucid, court, extracted, reviews)}, mentioned_only rows).

    1. template lines: a line repeated (fuzzily) in documents of 2+ cases, or in 2+ documents of one case without a name
       filling it, is form text: it never gives a role by itself; it is paired with its nearest value line by layout and
       its meaning (party / lawyer / judge / other field) comes from the zero-shot model
    2. participant vs mentioned: a name that fills its line (few other words) fills a slot; a name inside a prose line is
       only mentioned (mentioned_only.jsonl) unless a form label points to it
    3. one role per name per case from the combined evidence of every document of the case; conflicts go to review
    4. an organisation is a firm only when attached to a lawyer (same block, or a shared email domain)
    5. docket roles of the same case override document guesses"""
    tc = cfg["template"]
    type_of = cfg["label_types"]
    min_conf = cfg["min_entity_confidence"]
    W = cfg["one_role"]
    docs = {}
    for filename, data, kind in items:
        blocks = doc_blocks(data, kind)
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
        docs[filename] = {"kind": kind, "blocks": blocks, "ucid": ucid, "court": court, "reviews": reviews}
        extractor.remember_lines(filename, [norm_line(ln) for b in blocks for ln in b["lines"]], ucid)
    found = extractor.run([b["text"] for d in docs.values() for b in d["blocks"]])
    known = extractor.known_lines()
    counts = template_lines(known, tc["line_similarity"])
    grams = ngram_stats(known)
    name_labels = set(type_of) | {"mentioned"}
    for d in docs.values():
        for b in d["blocks"]:
            starts, pos = [], 0
            for ln in b["lines"]:
                starts.append(pos)
                pos += len(ln) + 1
            b["starts"], b["ents"] = starts, found.get(b["text"], [])
            b["tmpl"] = []
            for i, ln in enumerate(b["lines"]):
                n_docs, n_cases = counts.get(norm_line(ln), (0, 0))
                share, g_cases = ngram_template(norm_line(ln), grams, tc["min_documents"])
                if share >= tc["ngram_share"]:
                    n_docs, n_cases = max(n_docs, tc["min_documents"]), max(n_cases, g_cases)
                letters = sum(c.isalpha() for c in ln)
                covered = sum(sum(c.isalpha() for c in e["text"]) for e in b["ents"]
                              if e["label"] in name_labels and (e.get("confidence") or 1) >= min_conf
                              and e.get("start") is not None and starts[i] <= e["start"] < starts[i] + len(ln) + 1)
                b["tmpl"].append(n_docs >= tc["min_documents"] and (n_cases >= 2 or covered * 2 < max(1, letters)))
    def has_name(b, j):
        """line j of block b holds a name the model found (any person / organisation label)"""
        lo, hi = b["starts"][j], b["starts"][j] + len(b["lines"][j]) + 1
        return any(e["label"] in name_labels and e.get("start") is not None and lo <= e["start"] < hi
                   and (e.get("confidence") or 1) >= cfg["contact_min_confidence"] for e in b["ents"])

    # pair each template label with its nearest value line that holds a name (same block first, then by layout)
    labels_seen = set()
    for d in docs.values():
        for b in d["blocks"]:
            b["fields"] = {}
        for bi, b in enumerate(d["blocks"]):
            for i, is_t in enumerate(b["tmpl"]):
                if (not is_t or len(re.findall(r"[A-Za-z]{2,}", b["lines"][i])) > tc["label_max_words"]
                        or b["lines"][i].rstrip().endswith(".")):
                    continue   # long or sentence-final repeated lines are boilerplate prose, not field labels
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
                    _, _, bj, j = min(cands)
                    d["blocks"][bj]["fields"].setdefault(j, []).append(b["lines"][i].strip())
                    labels_seen.add(b["lines"][i].strip())
    field_cls = extractor.classify(sorted(labels_seen), "field", labels_of(cfg["field_labels"]))
    role_of_label = {"attorney": "counsel", "judge": "judge", "plaintiff": "party", "defendant": "party", "party": "party"}
    label_ner = extractor.run(sorted(labels_seen))
    from rapidfuzz import fuzz
    seen = sorted(labels_seen)
    ner_best = {}
    for lab_text in seen:
        best = None
        for e in label_ner.get(lab_text, []):
            r_ = role_of_label.get(e["label"])
            if r_ and (e.get("confidence") or 0) >= tc["label_min_confidence"] and (best is None or e["confidence"] > best[1]):
                best = (r_, e["confidence"])
        ner_best[lab_text] = best
    ner_roles = {}
    for lab_text in seen:
        span = {}
        for e in label_ner.get(lab_text, []):
            r_ = role_of_label.get(e["label"])
            if r_ and (e.get("confidence") or 0) >= tc["label_min_confidence"]:
                span[r_] = max(span.get(r_, 0), len(e["text"]))
        ner_roles[lab_text] = span
    for lab_text in seen:
        group = [x for x in seen if line_score(norm_line(x), norm_line(lab_text)) >= tc["line_similarity"] * 100]
        span = {}
        for x in group:
            for r_, v in ner_roles[x].items():
                span[r_] = max(span.get(r_, 0), v)
        cls = field_cls.get(lab_text, ("other", None))
        if len(span) == 1:
            field_cls[lab_text] = (next(iter(span)), max(ner_best[x][1] for x in group if ner_best[x]))
        elif len(span) > 1:
            # the label names several roles: the classifier picks among them; else a role whose span clearly dominates
            # ("Defendant's Attorney"); else no field meaning
            top = sorted(span.items(), key=lambda kv: -kv[1])
            if cls[0] in span:
                field_cls[lab_text] = cls
            elif top[0][1] >= 1.5 * top[1][1]:
                field_cls[lab_text] = (top[0][0], None)
            else:
                field_cls[lab_text] = ("other", None)
    # occurrences of names on content lines, with their evidence
    occ = []
    contact_labels = set(cfg["contact_labels"])
    for f, d in docs.items():
        for b in d["blocks"]:
            filled = [ln for ln in b["lines"] if ln.strip()]
            b["prose"] = bool(filled) and sum(len(re.findall(r"[A-Za-z]{2,}", ln)) for ln in filled) / len(filled) >= tc["prose_block_words"]
            b["contact"] = sorted({e["label"] for e in b["ents"] if e["label"] in contact_labels
                                   and (e.get("confidence") or 0) >= cfg["contact_min_confidence"]})
            b["lawfirm"] = any(e["label"] == "law firm" and (e.get("confidence") or 0) >= min_conf for e in b["ents"])
            cmin = cfg["contact_min_confidence"]
            emails = sorted({x.lower() for x in EMAIL_RE.findall(b["text"])}
                            | {e["text"].lower() for e in b["ents"] if e["label"] == "email" and "@" in e["text"] and (e.get("confidence") or 0) >= cmin})
            phones = sorted(x for x in ({re.sub(r"\D", "", x) for x in PHONE_RE.findall(b["text"])}
                                        | {re.sub(r"\D", "", e["text"]) for e in b["ents"] if e["label"] == "phone" and (e.get("confidence") or 0) >= cmin})
                            if len(x) >= 10)
            spans = {}
            for e in b["ents"]:
                if e["label"] not in name_labels or e.get("start") is None:
                    continue
                text = [x for x in e["text"].split("\n") if x.strip()][-1].strip(" ,;:.") if e["text"].strip() else ""
                off = e["text"].rfind(text) if text else 0
                st = e["start"] + max(off, 0)
                line = max([i for i, s0 in enumerate(b["starts"]) if s0 <= st] or [0])
                if b["tmpl"][line] or not re.search(r"[A-Za-z]{3}", text):
                    continue   # on a form line, or no word of 3+ letters (OCR debris)
                if counts.get(norm_line(text), (0, 0))[1] >= 2:
                    continue   # the string is itself form text repeated across cases ("Defendant"), not a name
                k = (line, norm_line(text).replace("#", "").strip())
                sp = spans.setdefault(k, {"name": text, "labels": {}, "start": st, "end": st + len(text), "line": line})
                sp["labels"][e["label"]] = max(sp["labels"].get(e["label"], 0), e.get("confidence") or 1.0)
            for (line, key), sp in spans.items():
                where = {"page": b["page"], "bbox": b["bbox"], "block_confidence": b["conf"], "block_class": b["class"],
                         "json_path": b["path"]}
                best = max(sp["labels"].values())
                if best < min_conf:
                    d["reviews"].append({"file": f, "ucid": d["ucid"], "name": sp["name"], "labels": sp["labels"],
                                         **{k_: v for k_, v in where.items() if v is not None}, "reason": f"confidence below {min_conf}"})
                    continue
                ln = b["lines"][line]
                other_words = len(re.findall(r"[A-Za-z]{2,}", ln)) - len(re.findall(r"[A-Za-z]{2,}", sp["name"]))
                fields = b["fields"].get(line, [])
                ftypes = [field_cls.get(x, ("other", None))[0] for x in fields]
                paired = any(t in ("party", "counsel", "judge") for t in ftypes)
                occ.append({"file": f, "ucid": d["ucid"], "name": sp["name"], "key": key, "labels": dict(sp["labels"]),
                            "fields": fields, "ftypes": ftypes, "block": b, "where": where, "emails": emails,
                            "phones": phones, "span": (sp["start"], sp["end"]), "line_text": ln.strip(),
                            "value_line": paired or (other_words <= tc["prose_min_other_words"] and not b["prose"])})
    # second model pass on each value line alone: adds judge / attorney evidence (the block context can hide it)
    line_found = extractor.run(sorted({o["line_text"] for o in occ if o["value_line"]}))
    for o in occ:
        if not o["value_line"]:
            continue
        for e in line_found.get(o["line_text"], []):
            if e["label"] in ("judge", "attorney") and norm_line(e["text"]).replace("#", "").strip() == o["key"]:
                o["labels"][e["label"]] = max(o["labels"].get(e["label"], 0), e.get("confidence") or 0)
    kinds = {}
    for o in occ:
        if o["name"] not in kinds:
            lab = max(o["labels"], key=o["labels"].get)
            kinds[o["name"]] = kind_of(o["name"], lab)
    # votes per occurrence
    for o in occ:
        kind = kinds[o["name"]]
        v = {}
        for lab, c in o["labels"].items():
            t = type_of.get(lab)
            if t:
                w = W["person_party_label_weight"] if (kind == "PERSON" and t == "party") else 1.0
                v[t] = v.get(t, 0) + c * w
        for ft in o["ftypes"]:
            if ft in ("party", "counsel", "judge"):
                v[ft] = v.get(ft, 0) + W["label_weight"]
        if kind == "PERSON":
            n_people = len({x["name"] for x in occ if x["block"] is o["block"] and kinds[x["name"]] == "PERSON"})
            if ((o["emails"] or o["phones"] or o["block"]["contact"]) and n_people <= W["contact_max_people"]
                    and "judge" not in o["ftypes"]):
                v["counsel"] = v.get("counsel", 0) + W["contact_bonus"]
            if o["block"]["lawfirm"] and "judge" not in o["ftypes"]:
                v["counsel"] = v.get("counsel", 0) + W["firm_bonus"]
            v["counsel"] = v.get("counsel", 0) + v.pop("firm", 0)
        o["votes"], o["kind"] = v, kind
    counsel_blocks, counsel_domains = set(), {}
    for o in occ:
        if o["kind"] == "PERSON" and o["value_line"] and o["votes"].get("counsel", 0) > max(o["votes"].get("party", 0), o["votes"].get("judge", 0)):
            counsel_blocks.add(id(o["block"]))
            counsel_domains.setdefault(o["ucid"], set()).update(x.split("@", 1)[1] for x in o["emails"] if "@" in x)
    for o in occ:
        if o["kind"] != "ORGANISATION":
            continue
        v = o["votes"]
        v.pop("judge", None)
        lawyer_side = v.pop("counsel", 0) + v.pop("firm", 0)
        attached = id(o["block"]) in counsel_blocks or bool(
            {x.split("@", 1)[1] for x in o["emails"] if "@" in x} & counsel_domains.get(o["ucid"], set()))
        if attached:
            v["firm"] = lawyer_side   # an organisation is a firm only when attached to a lawyer
    if debug is not None:
        debug.extend({k: o[k] for k in ("file", "name", "labels", "fields", "ftypes", "votes", "kind", "value_line")}
                     | {"line": o["block"]["lines"][next(i for i, s0 in reversed(list(enumerate(o["block"]["starts"]))) if s0 <= o["span"][0])][:80]}
                     for o in occ)
        debug.append({"field_classes": field_cls})
    # one role per name per case
    agg = {}
    for o in occ:
        a_ = agg.setdefault((o["ucid"], o["key"]), {"votes": {}, "occ": [], "participant": False})
        a_["occ"].append(o)
        if o["value_line"] and (o["kind"] in ("PERSON", "ORGANISATION") or any(t in ("party", "counsel", "judge") for t in o["ftypes"])):
            a_["participant"] = True
            for t, w in o["votes"].items():
                a_["votes"][t] = a_["votes"].get(t, 0) + w
    out = {f: (d["ucid"], d["court"], [], d["reviews"]) for f, d in docs.items()}
    mentioned = []
    for (ucid, key), a_ in sorted(agg.items(), key=lambda kv: kv[0]):
        first = a_["occ"][0]
        dock = (docket_roles.get(ucid) or {}).get(key)
        vs = {t: w for t, w in a_["votes"].items() if t in ("party", "counsel", "judge", "firm") and w > 0}
        if dock:
            decided, why = dock, "docket JSON of the same case"
        elif not a_["participant"] or not vs:
            for o in a_["occ"]:
                mentioned.append({"file": o["file"], "ucid": ucid, "name": o["name"], "model_labels": o["labels"],
                                  **{k: v for k, v in o["where"].items() if v is not None},
                                  "reason": "named only inside text, never filling a participant slot"})
            continue
        else:
            tot = sum(vs.values())
            decided = max(sorted(vs), key=lambda t: vs[t])
            if vs[decided] < W["min_share"] * tot:
                out[first["file"]][3].append({"file": first["file"], "ucid": ucid, "name": first["name"],
                                              "votes": {t: round(w, 3) for t, w in sorted(vs.items())},
                                              "reason": "conflicting roles for this name in this case"})
                continue
            why = "form label + model evidence over all documents of the case"
        for o in a_["occ"]:
            if not o["value_line"] and not dock:
                mentioned.append({"file": o["file"], "ucid": ucid, "name": o["name"], "model_labels": o["labels"],
                                  **{k: v for k, v in o["where"].items() if v is not None},
                                  "reason": f"named inside text (also a {decided} of this case elsewhere)"})
                continue
            b = o["block"]
            best = max(o["labels"], key=o["labels"].get)
            base = {"document": [True], "document_file": [o["file"]], "model_label": [best],
                    # the reader's person / organisation decision is passed on only when structure supports it:
                    # a form label points to the name, or it is a firm attached to a lawyer
                    "reader_kind": [o["kind"]] if o["kind"] and (any(t in ("party", "counsel", "judge") for t in o["ftypes"])
                                                                 or decided == "firm") else [],
                    "confidence": [o["labels"][best]], "field": o["fields"][:2], "role_decision": [why]}
            for k, v in o["where"].items():
                if v is not None:
                    base[k] = [",".join(map(str, v)) if isinstance(v, list) else v]
            if decided == "judge":
                ctx = {**base, "docket": [True], "docket_title": [True]}
            elif decided == "party":
                side = cfg["label_sides"].get(max((l for l in o["labels"] if l in cfg["label_sides"]), key=o["labels"].get, default="party"), "other")
                ctx = {**base, "counsel_raw": [], "firm_raw": [], "role_raw": [best], "side": [side]}
            elif decided == "counsel":
                people = {x["name"] for x in occ if x["block"] is b and x["kind"] == "PERSON"}
                firms = sorted({x["name"] for x in occ if x["block"] is b and x["kind"] == "ORGANISATION" and x["votes"].get("firm")})
                one = len(people) <= 1
                ctx = {**base, "emails": o["emails"] if one else [], "phones": o["phones"] if one else [], "addresses": [],
                       "offices": [], "firm_raw": firms, "party_raw": [], "party_side": [], "bar": [], "rawinfo_staff": []}
            else:
                ctx = {**base, "addresses": [], "phones": o["phones"],
                       "domains": sorted({x.split("@", 1)[1] for x in o["emails"] if "@" in x})}
            out[o["file"]][2].append((o["name"], decided, ctx, b["text"], o["span"]))
    return out, mentioned

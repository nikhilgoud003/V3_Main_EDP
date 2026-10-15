"""Input readers for Layer A, chosen by the file's STRUCTURE (never its name):

  docket JSON    a dict with "ucid" and "parties" / "docket"        -> v4_run_ctx.extract_mentions_from_json
  OCR document   a dict with "pages": [{"blocks": [{"text", ...}]}]  -> read_document()
  anything else  any JSON                                            -> read_generic() (every text field + its JSON path)

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
FILE_RE = re.compile(r"^([a-z]{2,6});;(\d{1,2})-(\d{2})-([a-z]{2,5})-(\d{3,6})", re.I)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\(?\b\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")


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
        return hashlib.sha256(f"{self.mode}|{text}".encode()).hexdigest()

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

    def save(self):
        tmp = self.cache_path.with_name(self.cache_path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for k in sorted(self.cache):
                f.write(json.dumps(self.cache[k]) + "\n")
        tmp.replace(self.cache_path)


def text_items_document(data):
    """[(text, where)] for every OCR block: where = {page, bbox, block_confidence, block_class}."""
    out = []
    for p in data.get("pages") or []:
        for b in (p.get("blocks") or []) if isinstance(p, dict) else []:
            t = (b.get("text") or "").strip() if isinstance(b, dict) else ""
            if t:
                out.append((t, {"page": p.get("page"), "bbox": b.get("bbox"), "block_confidence": b.get("confidence"),
                                "block_class": b.get("class")}))
    return out


def text_items_generic(data, path="$"):
    """[(text, where)] for every string field with letters, where = {json_path}."""
    out = []
    if isinstance(data, dict):
        for k in sorted(data, key=str):
            out += text_items_generic(data[k], f"{path}.{k}")
    elif isinstance(data, list):
        for i, v in enumerate(data):
            out += text_items_generic(v, f"{path}[{i}]")
    elif isinstance(data, str) and re.search(r"[A-Za-z]{2}", data):
        out.append((data.strip(), {"json_path": path}))
    return out


def read_text_file(data, filename, kind, extractor, cfg):
    """(ucid, court, extracted, reviews): extracted = [(raw_name, etype, ctx, sentence, (start, end))]."""
    items = text_items_document(data) if kind == "document" else text_items_generic(data)
    texts = [t for t, _ in items]
    f_court, f_case = case_from_filename(filename)
    t_case = case_from_text(texts)
    reviews = []
    case = f_case or t_case
    court = f_court or "unknown"
    if f_case and t_case and f_case != t_case:
        reviews.append({"file": filename, "reason": f"case number in the text ({t_case}) differs from the file name ({f_case})"})
    if not case:
        reviews.append({"file": filename, "reason": "no case number in the file name or the text"})
        case = "unknown-" + hashlib.sha256(filename.encode()).hexdigest()[:10]
    ucid = f"{court};;{case}"
    found = extractor.run(texts)
    extracted = []
    min_conf = cfg["min_entity_confidence"]
    type_of = cfg["label_types"]
    for text, where in items:
        ents = found.get(text, [])
        emails = sorted({e["text"].lower() for e in ents if e["label"] == "email"} | {x.lower() for x in EMAIL_RE.findall(text)})
        phones = sorted({re.sub(r"\D", "", e["text"]) for e in ents if e["label"] == "phone"}
                        | {re.sub(r"\D", "", x) for x in PHONE_RE.findall(text)})
        lawyers = [e for e in ents if type_of.get(e["label"]) == "counsel"]
        firms = [e for e in ents if type_of.get(e["label"]) == "firm"]
        for e in ents:
            etype = type_of.get(e["label"])
            if not etype:
                continue
            conf = e.get("confidence")
            base = {"document": [True], "source_kind": [kind], "model_label": [e["label"]],
                    "confidence": [conf] if conf is not None else []}
            for k, v in where.items():
                if v is not None:
                    base[k] = [",".join(map(str, v)) if isinstance(v, list) else v]
            if conf is not None and conf < min_conf:
                reviews.append({"file": filename, "ucid": ucid, "name": e["text"], "label": e["label"], "confidence": conf,
                                **{k: v for k, v in where.items() if v is not None}, "reason": f"confidence below {min_conf}"})
                continue
            span = (e["start"], e["end"]) if e.get("start") is not None and e.get("end") is not None else None
            if etype == "judge":
                ctx = {**base, "docket": [True], "docket_title": [True]}
            elif etype == "party":
                side = cfg["label_sides"].get(e["label"], "other")
                ctx = {**base, "counsel_raw": [], "firm_raw": [], "role_raw": [e["label"]], "side": [side]}
            elif etype == "counsel":
                one = len(lawyers) == 1   # block-level contact details belong to the lawyer only when there is one
                ctx = {**base, "emails": emails if one else [], "phones": phones if one else [], "addresses": [],
                       "offices": [], "firm_raw": [f["text"] for f in firms] if one else [], "party_raw": [],
                       "party_side": [], "bar": [], "rawinfo_staff": []}
            else:
                ctx = {**base, "addresses": [], "phones": phones if len(firms) == 1 else [],
                       "domains": sorted({x.split("@", 1)[1] for x in emails}) if len(firms) == 1 else []}
            extracted.append((e["text"], etype, ctx, text, span))
    return ucid, court, extracted, reviews

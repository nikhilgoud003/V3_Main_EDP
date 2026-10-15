#!/usr/bin/env python3
"""V3 knowledge graph: RDF from the V3 export (+ Layer B redirects and related links) into a Tentris store.

Vocabulary: the SCALES PACER terms V3_overtime used (pacer: = http://scales-kg.org/pacer#, ids under http://scales-kg.org/id/<kind>/<id>,
skos: names). Terms V3 adds live in a separate namespace v3: (http://scales-kg.org/v3#):
  v3:Attorney / v3:AttorneyMention / v3:hasSCID      counsel (V3_overtime emitted no counsel)
  v3:mergedInto, v3:MergedRecord                     Layer B merge links (old ID -> surviving ID)
  v3:officeLink, v3:parentOrSubsidiaryLink            Layer B related organisations
  v3:plaintiffParty / v3:defendantParty / v3:otherParty   side of a party in one case
  v3:hasJudge, v3:hasAttorney, v3:represents         case-judge, case-attorney, attorney-party links
  v3:isOrganisation                                  the organisation flag of parties and counsel
  v3:plaintiffSideCount / v3:defendantSideCount / v3:otherSideCount, v3:sourceDocument, v3:UnresolvedJudgeRecord

Incremental loads are exact: the new export's triples are compared with the last loaded state; removed triples (a changed
display name, a merged-away ID) are deleted with SPARQL DELETE DATA and new ones added with INSERT DATA on the live store.
Every load is recorded in <data>/load_log.jsonl. The store is served on 127.0.0.1 only, on a free port chosen at start
(never 9080), written to <store>/.tentris_port.

Usage (all paths are arguments; nothing about the server is stored in the repo):
  python v3_kg.py start  --store DIR [--port N]
  python v3_kg.py stop   --store DIR
  python v3_kg.py emit   --export DIR --layer-b DIR --out FILE.nt
  python v3_kg.py build  --export DIR --layer-b DIR --store DIR [--batch-id ID]     (new, empty store; served after)
  python v3_kg.py add    --export DIR --layer-b DIR --store DIR [--batch-id ID]     (delta into the live store)
  python v3_kg.py query  --store DIR [--example N | --sparql TEXT]
  python v3_kg.py counts --export DIR --store DIR
  python v3_kg.py prove  --export-1 DIR --export-2 DIR --layer-b-1 DIR --layer-b-2 DIR --work DIR
"""
import argparse
import collections
import datetime
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

PACER = "http://scales-kg.org/pacer#"
ID = "http://scales-kg.org/id/"
V3 = "http://scales-kg.org/v3#"
SKOS = "http://www.w3.org/2004/02/skos/core#"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
XSD = "http://www.w3.org/2001/XMLSchema#"

KIND = {"SJ": "judge", "SP": "party", "SC": "counsel", "SF": "firm", "SJU": "judge_unresolved"}
CLASS = {"judge": PACER + "Judge", "party": PACER + "Party", "firm": PACER + "LawFirm", "counsel": V3 + "Attorney",
         "judge_unresolved": V3 + "UnresolvedJudgeRecord"}
ID_PRED = {"judge": PACER + "hasSJID", "party": PACER + "hasSPID", "firm": PACER + "hasSFID", "counsel": V3 + "hasSCID",
           "judge_unresolved": V3 + "hasRecordId"}
MENTION_CLASS = {"judge": PACER + "JudgeMention", "party": PACER + "PartyMention", "firm": PACER + "FirmMention",
                 "counsel": V3 + "AttorneyMention"}
SIDE = {"plaintiff-side": "plaintiff", "defendant-side": "defendant"}
CASE_PARTY = {"plaintiff": V3 + "plaintiffParty", "defendant": V3 + "defendantParty", "other": V3 + "otherParty"}
RELATION = {"office": V3 + "officeLink", "parent_or_subsidiary": V3 + "parentOrSubsidiaryLink"}
ENTITY_LINKS = {PACER + "resolvedTo", V3 + "hasJudge", V3 + "hasAttorney", PACER + "hasCounselOffice", V3 + "represents",
                V3 + "mergedInto", V3 + "officeLink", V3 + "parentOrSubsidiaryLink", PACER + "mentionsIn",
                *CASE_PARTY.values()}
TENTRIS = os.environ.get("TENTRIS_BIN", str(Path.home() / ".local/bin/tentris"))
FORBIDDEN_PORTS = {9080}


# ---------------------------------------------------------------- RDF terms
def iri(x):
    return f"<{x}>"


def lit(s, dtype=None):
    s = str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return f'"{s}"' + (f"^^<{XSD}{dtype}>" if dtype else "")


def entity_iri(eid):
    """ent_SJ_000123 -> <http://scales-kg.org/id/judge/SJ000123> (the V3_overtime id pattern)."""
    m = re.match(r"ent_([A-Z]+)_(\d+)$", eid or "")
    if not m or m.group(1) not in KIND:
        raise ValueError(f"unexpected entity id {eid!r}")
    kind = KIND[m.group(1)]
    return kind, f"{ID}{kind}/{m.group(1)}{m.group(2)}", f"{m.group(1)}{m.group(2)}"


def node(kind, key):
    return f"{ID}{kind}/{urllib.parse.quote(str(key), safe='')}"


def read_jsonl(p):
    p = Path(p)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []


def collapse(redirects):
    to = {r["old_id"]: r["surviving_id"] for r in redirects}

    def final(i):
        seen = set()
        while i in to and i not in seen:
            seen.add(i)
            i = to[i]
        return i
    return {i: final(i) for i in to}


# ---------------------------------------------------------------- emit
def emit(export_dir, layer_b_dir):
    """Set of N-Triples lines for one V3 export + its Layer B folder."""
    export_dir, layer_b_dir = Path(export_dir), Path(layer_b_dir)
    ents = read_jsonl(export_dir / "entities.jsonl")
    ments = read_jsonl(export_dir / "mentions.jsonl")
    redirects = read_jsonl(layer_b_dir / "redirects.jsonl")
    links = read_jsonl(layer_b_dir / "related_links.jsonl")
    final_of = collapse(redirects)
    fin = lambda i: final_of.get(i, i)
    T = set()
    add = lambda s, p, o: T.add(f"{iri(s)} {iri(p)} {o} .")
    known = {}
    for e in ents:
        kind, u, local = entity_iri(e["entity_id"])
        known[e["entity_id"]] = u
        add(u, RDF_TYPE, iri(CLASS[kind]))
        name = " ".join(str(e.get("canonical_name") or "").split())
        add(u, PACER + "hasName", lit(name))
        add(u, SKOS + "prefLabel", lit(name))
        add(u, ID_PRED[kind], lit(local))
        for v in e.get("name_variants") or []:
            if v and v.lower() != name.lower():
                add(u, SKOS + "altLabel", lit(v))
        for c in e.get("courts") or []:
            add(u, PACER + "hasCode", lit(c))
        if "is_organisation" in e:
            add(u, V3 + "isOrganisation", lit("true" if e["is_organisation"] else "false", "boolean"))
        if kind == "party":
            for k, p in (("plaintiff_side", "plaintiffSideCount"), ("defendant_side", "defendantSideCount"),
                         ("other_side", "otherSideCount")):
                add(u, V3 + p, lit(int(e.get(k) or 0), "integer"))
    for m in ments:
        if not m.get("entity_id") or not m.get("mention_id"):
            continue    # pending docket judge names have no entity yet
        eid = fin(m["entity_id"])
        if eid not in known:
            continue
        t = m.get("entity_type")
        mu = node("mention", m["mention_id"])
        cu = node("case", m.get("case") or "unknown")
        add(cu, RDF_TYPE, iri(PACER + "Case"))
        add(cu, PACER + "hasUcid", lit(m.get("case") or ""))
        if m.get("court"):
            add(cu, PACER + "hasCode", lit(m["court"]))
        add(mu, RDF_TYPE, iri(MENTION_CLASS[t]))
        add(mu, PACER + "hasName", lit(" ".join(str(m.get("raw_name") or "").split())))
        add(mu, PACER + "mentionsIn", iri(cu))
        add(mu, PACER + "resolvedTo", iri(known[eid]))
        ctx = m.get("context") or {}
        for f in ctx.get("document_file") or []:
            add(mu, V3 + "sourceDocument", lit(f))
        if t == "judge":
            add(cu, V3 + "hasJudge", iri(known[eid]))
        elif t == "party":
            sides = [SIDE.get(s, "other") for s in (m.get("side") or [])] or ["other"]
            for sd in sorted(set(sides)):
                add(mu, PACER + "hasRole", lit(sd))
                add(cu, CASE_PARTY[sd], iri(known[eid]))
        elif t == "counsel":
            add(cu, V3 + "hasAttorney", iri(known[eid]))
            for pid in ctx.get("party_eids") or []:
                if fin(pid) in known:
                    add(known[eid], V3 + "represents", iri(known[fin(pid)]))
        elif t == "firm":
            add(cu, PACER + "hasCounselOffice", iri(known[eid]))
    for old, new in sorted(final_of.items()):
        if new not in known:
            continue
        kind, ou, local = entity_iri(old)
        add(ou, RDF_TYPE, iri(V3 + "MergedRecord"))
        add(ou, ID_PRED[kind], lit(local))
        add(ou, V3 + "mergedInto", iri(known[new]))
    for r in links:
        a, b = fin(r["id_a"]), fin(r["id_b"])
        if a in known and b in known and r.get("relation") in RELATION:
            add(known[a], RELATION[r["relation"]], iri(known[b]))
    return T


def validate(triples):
    """It parses; every entity has a type and a name; no dangling IDs. Returns (ok, report)."""
    import rdflib
    g = rdflib.Graph()
    g.parse(data="\n".join(sorted(triples)), format="nt")
    typed = {str(s) for s, o in g.subject_objects(rdflib.URIRef(RDF_TYPE))}
    named = {str(s) for s in g.subjects(rdflib.URIRef(PACER + "hasName"))}
    entity_classes = {rdflib.URIRef(c) for c in CLASS.values()}
    entities = {str(s) for s, o in g.subject_objects(rdflib.URIRef(RDF_TYPE)) if o in entity_classes}
    no_name = sorted(entities - named)
    dangling = sorted({str(o) for s, p, o in g if str(p) in ENTITY_LINKS and isinstance(o, rdflib.URIRef) and str(o) not in typed})
    untyped_links = sorted({str(s) for s, p, o in g if str(p) in ENTITY_LINKS and str(s) not in typed})
    by_class = collections.Counter(str(o).split("#")[-1] for s, o in g.subject_objects(rdflib.URIRef(RDF_TYPE)))
    rep = {"triples": len(g), "parsed": True, "entities": len(entities), "entities_without_name": len(no_name),
           "dangling_objects": len(dangling), "untyped_subjects": len(untyped_links), "by_class": dict(sorted(by_class.items())),
           "examples": {"no_name": no_name[:5], "dangling": dangling[:5], "untyped": untyped_links[:5]}}
    return not no_name and not dangling and not untyped_links, rep


# ---------------------------------------------------------------- store
def free_port():
    while True:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
        s.close()
        if p not in FORBIDDEN_PORTS:
            return p


def port_of(store):
    f = Path(store) / ".tentris_port"
    return int(f.read_text()) if f.exists() else None


def running(store):
    pf = Path(store) / ".tentris_pid"
    if not pf.exists():
        return False
    try:
        os.kill(int(pf.read_text()), 0)
        return True
    except (OSError, ValueError):
        return False


def start(store, port=None):
    store = Path(store).expanduser()
    if running(store):
        return port_of(store)
    if not store.exists():
        store.mkdir(parents=True)
        subprocess.run([TENTRIS, "-s", str(store / "db"), "init"], check=True, capture_output=True)
    port = int(port) if port else free_port()
    if port in FORBIDDEN_PORTS:
        sys.exit(f"port {port} is reserved for the old store; choose another")
    log = open(store / "tentris.log", "a")
    p = subprocess.Popen([TENTRIS, "-s", str(store / "db"), "serve", f"127.0.0.1:{port}"], stdout=log, stderr=log,
                         start_new_session=True)
    (store / ".tentris_pid").write_text(str(p.pid))
    (store / ".tentris_port").write_text(str(port))
    for _ in range(100):
        try:
            sparql(store, "ASK { ?s ?p ?o }")
            return port
        except Exception:
            time.sleep(0.2)
    sys.exit("Tentris did not answer; see tentris.log in the store folder")


def stop(store):
    store = Path(store).expanduser()
    pf = store / ".tentris_pid"
    if running(store):
        pid = int(pf.read_text())
        os.kill(pid, signal.SIGTERM)
        for _ in range(100):
            try:
                os.kill(pid, 0)
                time.sleep(0.1)
            except OSError:
                break
    if pf.exists():
        pf.unlink()


def _post(store, route, field, text, accept):
    port = port_of(store)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/{route}", urllib.parse.urlencode({field: text}).encode(),
                                 {"Accept": accept, "Content-Type": "application/x-www-form-urlencoded"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())


def sparql(store, q):
    return _post(store, "sparql", "query", q, "application/sparql-results+json")


def update(store, u):
    return _post(store, "update", "update", u, "application/json")


def rows(res):
    return [{k: v["value"] for k, v in b.items()} for b in res["results"]["bindings"]]


def count(store):
    return int(rows(sparql(store, "SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }"))[0]["n"])


def store_triples(store):
    """Every triple of a store as N-Triples lines (from `tentris dump`, so the store must be stopped)."""
    store = Path(store).expanduser()
    was = running(store)
    port = port_of(store)
    if was:
        stop(store)
    out = subprocess.run([TENTRIS, "-s", str(store / "db"), "dump"], capture_output=True, text=True, check=True).stdout
    if was:
        start(store, port)
    return {l.strip() for l in out.splitlines() if l.strip()}


def apply_delta(store, delete, insert, chunk=5000):
    stats = {"inserted": 0, "removed": 0}
    for verb, items in (("DELETE", sorted(delete)), ("INSERT", sorted(insert))):
        for i in range(0, len(items), chunk):
            r = update(store, f"{verb} DATA {{\n" + "\n".join(items[i:i + chunk]) + "\n}")
            stats["inserted"] += r.get("statements_inserted", 0)
            stats["removed"] += r.get("statements_removed", 0)
    return stats


# ---------------------------------------------------------------- build / add
def state_file(store):
    return Path(store).expanduser() / "state_loaded.nt"


def log_load(store, row):
    lf = Path(store).expanduser().parent / "load_log.jsonl"
    with open(lf, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return lf


def write_nt(path, triples):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(sorted(triples)) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def build(export_dir, layer_b_dir, store, batch_id=None, port=None):
    store = Path(store).expanduser()
    if store.exists() and any(store.iterdir()):
        sys.exit(f"{store} is not empty: build makes a NEW store (use add for a live store)")
    t0 = time.time()
    triples = emit(export_dir, layer_b_dir)
    ok, rep = validate(triples)
    if not ok:
        sys.exit("validation failed: " + json.dumps(rep, indent=1))
    store.mkdir(parents=True, exist_ok=True)
    nt = store / "build.nt"
    write_nt(nt, triples)
    subprocess.run([TENTRIS, "-s", str(store / "db"), "load", str(nt)], check=True, capture_output=True)
    nt.unlink()
    write_nt(state_file(store), triples)
    p = start(store, port)
    total = count(store)
    row = {"batch_id": batch_id or "build", "kind": "full build", "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "export": Path(export_dir).name, "triples_emitted": len(triples), "inserted": total, "removed": 0,
           "store_total": total, "seconds": round(time.time() - t0, 2), "validation": rep}
    lf = log_load(store, row)
    print(f"built: {total} triples in {row['seconds']}s; serving on 127.0.0.1:{p}; log {lf}")
    return row


def add(export_dir, layer_b_dir, store, batch_id=None):
    store = Path(store).expanduser()
    if not running(store):
        sys.exit("the store is not running: start it first")
    t0 = time.time()
    new = emit(export_dir, layer_b_dir)
    ok, rep = validate(new)
    if not ok:
        sys.exit("validation failed: " + json.dumps(rep, indent=1))
    old = {l for l in open(state_file(store), encoding="utf-8") if l.strip()} if state_file(store).exists() else set()
    old = {l.rstrip("\n") for l in old}
    insert, delete = new - old, old - new
    before = count(store)
    stats = apply_delta(store, delete, insert)
    after = count(store)
    write_nt(state_file(store), new)
    row = {"batch_id": batch_id or hashlib.sha256("".join(sorted(insert | delete)).encode()).hexdigest()[:12],
           "kind": "incremental", "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "export": Path(export_dir).name, "triples_emitted": len(new), "delta_insert": len(insert), "delta_delete": len(delete),
           "inserted": stats["inserted"], "removed": stats["removed"], "store_before": before, "store_total": after,
           "seconds": round(time.time() - t0, 2), "validation": {k: rep[k] for k in ("triples", "entities", "dangling_objects")}}
    lf = log_load(store, row)
    print(f"added batch {row['batch_id']}: +{stats['inserted']} -{stats['removed']} triples "
          f"({before} -> {after}) in {row['seconds']}s; log {lf}")
    return row


# ---------------------------------------------------------------- checks and examples
COUNT_CLASS = {"judge": "Judge", "party": "Party", "firm": "LawFirm", "counsel": "Attorney"}


def graph_counts(store):
    out = {}
    for t, c in COUNT_CLASS.items():
        ns = V3 if t == "counsel" else PACER
        out[t] = int(rows(sparql(store, f"SELECT (COUNT(DISTINCT ?e) AS ?n) WHERE {{ ?e a <{ns}{c}> }}"))[0]["n"])
    return out


def export_counts(export_dir):
    return dict(collections.Counter(e["entity_type"] for e in read_jsonl(Path(export_dir) / "entities.jsonl")
                                    if not e.get("unresolved")))


P = f"PREFIX pacer: <{PACER}> PREFIX v3: <{V3}> PREFIX skos: <{SKOS}> "


def examples(store):
    """Five example queries; the entity each one is about is picked by an earlier query (most cases / most lawyers...)."""
    out = []
    top_judge = rows(sparql(store, P + "SELECT ?j (COUNT(DISTINCT ?c) AS ?n) WHERE { ?c v3:hasJudge ?j } GROUP BY ?j ORDER BY DESC(?n) LIMIT 1"))
    if top_judge:
        j = top_judge[0]["j"]
        q = P + f"SELECT ?name ?ucid WHERE {{ <{j}> pacer:hasName ?name . ?c v3:hasJudge <{j}> . ?c pacer:hasUcid ?ucid }} ORDER BY ?ucid"
        r = rows(sparql(store, q))
        out.append({"query": "1. all cases of one judge (the judge with most cases)", "sparql": q, "n_results": len(r), "sample": r[:5]})
    top_party = rows(sparql(store, P + "SELECT ?p (COUNT(DISTINCT ?a) AS ?n) WHERE { ?a v3:represents ?p } GROUP BY ?p ORDER BY DESC(?n) LIMIT 1"))
    if top_party:
        pt = top_party[0]["p"]
        q = P + f"SELECT ?party ?lawyer WHERE {{ <{pt}> pacer:hasName ?party . ?a v3:represents <{pt}> . ?a pacer:hasName ?lawyer }} ORDER BY ?lawyer"
        r = rows(sparql(store, q))
        out.append({"query": "2. all lawyers for one party (the party with most lawyers)", "sparql": q, "n_results": len(r), "sample": r[:5]})
    q = P + "SELECT ?old ?survivor ?name WHERE { ?m v3:mergedInto ?e . ?m pacer:hasSJID ?old . ?e pacer:hasSJID ?survivor . ?e pacer:hasName ?name } ORDER BY ?old"
    r = rows(sparql(store, q))
    q2 = P + "SELECT (COUNT(?m) AS ?n) WHERE { ?m v3:mergedInto ?e }"
    out.append({"query": "3. entities merged by Layer B (judges listed; all types counted)", "sparql": q, "n_results": len(r),
                "all_merged_records": int(rows(sparql(store, q2))[0]["n"]), "sample": r[:5]})
    top_firm = rows(sparql(store, P + "SELECT ?f (COUNT(?o) AS ?n) WHERE { { ?f v3:officeLink ?o } UNION { ?o v3:officeLink ?f } ?f a pacer:LawFirm } GROUP BY ?f ORDER BY DESC(?n) LIMIT 1"))
    if top_firm:
        f_ = top_firm[0]["f"]
        q = P + (f"SELECT ?firm ?office WHERE {{ <{f_}> pacer:hasName ?firm . {{ <{f_}> v3:officeLink ?o }} UNION {{ ?o v3:officeLink <{f_}> }} "
                 f"?o pacer:hasName ?office }} ORDER BY ?office")
        r = rows(sparql(store, q))
        out.append({"query": "4. one firm's offices (the firm with most office links)", "sparql": q, "n_results": len(r), "sample": r[:5]})
    q = P + ("SELECT ?side (COUNT(*) AS ?links) WHERE { { ?c v3:plaintiffParty ?p BIND(\"plaintiff\" AS ?side) } UNION "
             "{ ?c v3:defendantParty ?p BIND(\"defendant\" AS ?side) } UNION { ?c v3:otherParty ?p BIND(\"other\" AS ?side) } } GROUP BY ?side ORDER BY ?side")
    r = rows(sparql(store, q))
    out.append({"query": "5. plaintiff vs defendant (case-party links by side)", "sparql": q, "n_results": len(r), "sample": r})
    return out


def prove(a):
    """(a) first batch + incremental second batch == fresh build of both at once; (b) re-loading the same batch adds 0;
    (c) entity counts by type == the export; (d) five example queries."""
    work = Path(a.work).expanduser()
    inc, fresh = work / "store", work / "store_fresh_check"
    rep = {}
    if not (inc.exists() and any(inc.iterdir())):
        build(a.export_1, a.layer_b_1, inc, "first batch")
    else:
        start(inc)
    if not running(inc):
        start(inc)
    loaded = [r for r in read_jsonl(work / "load_log.jsonl") if r.get("export") == Path(a.export_2).name]
    if not loaded:
        add(a.export_2, a.layer_b_2, inc, "second batch")
    r_b = add(a.export_2, a.layer_b_2, inc, "second batch (same batch again)")
    rep["b_reload_same_batch"] = {"inserted": r_b["inserted"], "removed": r_b["removed"], "pass": r_b["inserted"] == 0 and r_b["removed"] == 0}
    forced = apply_delta(inc, set(), emit(a.export_2, a.layer_b_2))
    rep["b_reinsert_every_triple"] = {"inserted": forced["inserted"], "pass": forced["inserted"] == 0}
    if fresh.exists():
        stop(fresh)
        shutil.move(str(fresh), str(work / f"store_fresh_check_old_{int(time.time())}"))
    fresh_parent_log = work / "load_log.jsonl"
    build(a.export_2, a.layer_b_2, fresh, "fresh build of both batches")
    t_inc, t_fresh = store_triples(inc), store_triples(fresh)
    rep["a_incremental_equals_fresh"] = {"incremental_triples": len(t_inc), "fresh_triples": len(t_fresh),
                                         "only_incremental": len(t_inc - t_fresh), "only_fresh": len(t_fresh - t_inc),
                                         "pass": t_inc == t_fresh}
    gc, ec = graph_counts(inc), export_counts(a.export_2)
    rep["c_counts"] = {"graph": gc, "export": ec, "pass": all(gc.get(k, 0) == ec.get(k, 0) for k in set(gc) | set(ec))}
    rep["d_examples"] = examples(inc)
    stop(fresh)
    (work / "proof.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: (v.get("pass") if isinstance(v, dict) else None) for k, v in rep.items() if k != "d_examples"}, indent=1))
    for x in rep["d_examples"]:
        print(f"\n{x['query']}: {x['n_results']} results" + (f" ({x['all_merged_records']} merged records in all)" if "all_merged_records" in x else ""))
        for s in x["sample"]:
            print("   ", s)
    print(f"\nproof written to {work / 'proof.json'}; load log {fresh_parent_log}")
    return rep


def main():
    ap = argparse.ArgumentParser(description="V3 knowledge graph (Tentris)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("start", "stop", "build", "add", "query", "counts", "emit", "prove"):
        s = sub.add_parser(name)
        if name in ("start", "stop", "build", "add", "query", "counts"):
            s.add_argument("--store", required=True)
        if name in ("emit", "build", "add"):
            s.add_argument("--export", required=True)
            s.add_argument("--layer-b", required=True)
        if name in ("build", "add"):
            s.add_argument("--batch-id")
        if name in ("start", "build"):
            s.add_argument("--port", type=int)
        if name == "emit":
            s.add_argument("--out", required=True)
        if name == "counts":
            s.add_argument("--export", required=True)
        if name == "query":
            s.add_argument("--example", type=int)
            s.add_argument("--sparql")
        if name == "prove":
            for k in ("--export-1", "--export-2", "--layer-b-1", "--layer-b-2", "--work"):
                s.add_argument(k, required=True)
    a = ap.parse_args()
    if a.cmd == "start":
        print(f"serving on 127.0.0.1:{start(a.store, a.port)}")
    elif a.cmd == "stop":
        stop(a.store)
        print("stopped")
    elif a.cmd == "emit":
        t = emit(a.export, a.layer_b)
        ok, rep = validate(t)
        write_nt(a.out, t)
        print(json.dumps(rep, indent=1))
        sys.exit(0 if ok else 1)
    elif a.cmd == "build":
        build(a.export, a.layer_b, a.store, a.batch_id, a.port)
    elif a.cmd == "add":
        add(a.export, a.layer_b, a.store, a.batch_id)
    elif a.cmd == "counts":
        gc, ec = graph_counts(a.store), export_counts(a.export)
        print(json.dumps({"graph": gc, "export": ec, "equal": gc == {k: ec.get(k, 0) for k in gc}}, indent=1))
    elif a.cmd == "query":
        if a.sparql:
            print(json.dumps(rows(sparql(a.store, a.sparql)), indent=1))
        else:
            ex = examples(a.store)
            for x in ex if not a.example else [ex[a.example - 1]]:
                print(f"{x['query']}: {x['n_results']} results\n  {x['sparql']}")
                for s in x["sample"]:
                    print("   ", s)
    elif a.cmd == "prove":
        rep = prove(a)
        sys.exit(0 if all(v.get("pass") for k, v in rep.items() if isinstance(v, dict) and "pass" in v) else 1)


if __name__ == "__main__":
    main()

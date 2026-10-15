#!/usr/bin/env python3
"""Per-document report for a Layer A output folder that read OCR / generic documents:
entities by type (with their names), mentioned_only names, and review rows.

Usage: python doc_report.py <layer A output folder> [--names]
"""
import collections
import json
import sys
from pathlib import Path


def rows(p):
    return [json.loads(l) for l in open(p)] if p.exists() else []


def main():
    out = Path(sys.argv[1]).expanduser()
    show = "--names" in sys.argv
    ents = {e["entity_id"]: e for e in rows(out / "entities.jsonl")}
    by_doc = collections.defaultdict(lambda: collections.defaultdict(set))
    for m in rows(out / "mentions.jsonl"):
        f = ((m.get("context") or {}).get("document_file") or [None])[0]
        if f:
            by_doc[f][m["entity_type"]].add(ents.get(m["entity_id"], {}).get("canonical_name") or m["raw_name"])
    mentioned = collections.defaultdict(list)
    for r in rows(out / "mentioned_only.jsonl"):
        mentioned[r["file"]].append(r["name"])
    review = collections.defaultdict(list)
    for r in rows(out / "document_review.jsonl"):
        review[r.get("file")].append(r)
    gate = collections.defaultdict(list)
    for r in rows(out / "gate_review.jsonl"):
        gate[r.get("ucid")].append(r["raw_name"])
    rep = {r["file"]: r for r in rows(out / "document_report.jsonl") if "file" in r}
    cal = [r for r in rows(out / "document_report.jsonl") if "calibration" in r]
    files = sorted(set(by_doc) | set(mentioned) | set(review) | set(rep))
    print(f"{'document':58s} {'judge':>5s} {'party':>5s} {'cnsl':>5s} {'firm':>5s} {'ment.':>5s} {'review':>6s} "
          f"{'gate P/O/rev':>13s} {'sec':>5s}")
    for f in files:
        t = by_doc[f]
        g = rep.get(f, {}).get("gate", {})
        gs = f"{g.get('PERSON', 0)}/{g.get('ORGANISATION', 0)}/{g.get('review', 0)}"
        print(f"{f[:58]:58s} {len(t['judge']):5d} {len(t['party']):5d} {len(t['counsel']):5d} {len(t['firm']):5d} "
              f"{len(set(mentioned[f])):5d} {len(review[f]):6d} {gs:>13s} {rep.get(f, {}).get('model_sec', 0):5.1f}")
        if show and rep.get(f, {}).get("gate_failed_examples"):
            print(f"    gate failed (review): {'; '.join(rep[f]['gate_failed_examples'])}")
        if show:
            for typ in ("judge", "party", "counsel", "firm"):
                if t[typ]:
                    print(f"    {typ:8s}: {'; '.join(sorted(t[typ]))}")
            if mentioned[f]:
                print(f"    mentioned only: {'; '.join(sorted(set(mentioned[f])))}")
            conf = [r for r in review[f] if "conflicting" in r.get("reason", "")]
            if conf:
                print(f"    review (role conflict): {'; '.join(f'{r['name']} {r['votes']}' for r in conf)}")
            other = [r for r in review[f] if "conflicting" not in r.get("reason", "")]
            if other:
                print(f"    review (other): {len(other)} rows, e.g. "
                      + "; ".join(f"{r.get('name', '')} ({r['reason'][:40]})" for r in other[:4]))
    if gate:
        print(f"\nname gate could not decide (not created): {sum(len(v) for v in gate.values())} names")
    for c in cal:
        print(f"\ncalibrated thresholds: {json.dumps(c['calibration'])}")
        print(f"time: model steps per document above; shared steps {c['shared_sec']}s; reader total {c['total_sec']}s")


if __name__ == "__main__":
    main()

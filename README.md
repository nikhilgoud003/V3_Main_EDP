# V4: entity resolution for PACER dockets

V4 reads PACER case JSON files and gives stable IDs to the **judges, parties, counsel and firms** named in them. It has two layers.

- **Layer A (`v4_run_ctx.py`): fast incremental linking.** One pass per file. Every mention is matched against an index of saved entities, keyed by type and surname / name, and merged only with context evidence:
  - Judges: same court plus a compatible name or title.
  - Person parties: shared counsel, firm or co-party.
  - Counsel: shared contact details.
  - Firms: shared address, phone or email domain.
  - Organisations: identical normalised names and acronyms.

  Re-running on the same output folder adds only new files (skipped by name + sha256). Old IDs never change. Every word list and threshold is in `v4_rules.yaml`, with a `source:` note for each.
- **Layer B (`v4_consolidate.py`): learned consolidation.** Runs after Layer A on its output folder and merges IDs that belong to the same real person or organisation. It is meant for cases like a judge split into "<first> <middle>" and "<first> <middle> <last>". A calibrated gradient-boosting model scores candidate ID pairs on generic features: name similarity, shared cases, divisions, co-occurring entities, party side, and competing candidates. The model was trained on the pilot answer key, the FJC judge list, and automatic name fragments, with a court-grouped split.
  - A pair merges when its score is >= T_high. T_high is set per pair kind so that held-out precision is >= 0.98.
  - Uncertain pairs go to a review queue.
  - Three fixed safety checks: never mix types; never join two different FJC judges; never join two different full first names with the same surname.
  - Redirects are append-only, so later runs keep earlier decisions. The surviving ID is always the oldest one.

  Settings are in `v4_consolidate.yaml`. The saved model and its card are in `models/`.

## Setup

Python 3.13 (3.12+ works for Layer A).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Reference data (FJC judge list, Census name files) is in `data/`; see `data/README.md`.

## Run

```bash
# 1. first run: --input is one or more .json files, a folder of .json files, or a .zip
python v4_run_ctx.py --input /path/to/case_json --output /path/to/run

# 2. add files later to the same output (processed files are skipped, old IDs stay)
python v4_run_ctx.py --input /path/to/new_case_json --output /path/to/run

# 3. Layer B with the saved model (also for a new court; no retraining)
python v4_consolidate.py --input /path/to/run --output /path/to/consolidated --model models/model.joblib
#    run it again on the same --output after step 2: only new or changed IDs are scored, earlier redirects are kept
python v4_consolidate.py --input /path/to/run --output /path/to/consolidated --model models/model.joblib

# 4. optional LLM review of the review queue (off by default)
python v4_consolidate.py --input /path/to/run --output /path/to/consolidated --model models/model.joblib \
    --llm local --cases /path/to/case_json          # Ollama; model set in v4_consolidate.yaml (llm.local_model)
ANTHROPIC_API_KEY=... python v4_consolidate.py ... --llm api   # the key is read from the environment only

# train a new Layer B model (needs the answer key folder; without it, FJC + fragments only)
python v4_consolidate.py --input /path/to/run --output /path/to/new_model_run --key /path/to/answer_key
```

Other Layer A options: `--rules my_rules.yaml`, `--order reverse` (file-order test), `--type-model gliner2` (optional model for undecided entity types; needs `pip install gliner2==2.1.0`).

**Layer A output** (in `--output`):
- `entities.jsonl`
- `mentions.jsonl`
- `state.json`
- `uncertain_pairs.jsonl`
- `pending_judges.jsonl` (docket judge names not yet promoted)
- `dropped_entities.jsonl`
- `type_cache.jsonl`

**Layer B output:**

| File | Contents |
|---|---|
| `entities_consolidated.jsonl` | one row per surviving ID, with `merged_from` |
| `redirects.jsonl` | old ID -> surviving ID, score, top reasons |
| `mentions_resolved.jsonl` | mention ID -> final entity ID |
| `review_queue.jsonl` | uncertain pairs, with both entity cards |
| `consolidation_state.json` | state for incremental runs |
| `report.md` | merges, review-queue sizes, time |
| `model_card.md` | features, weights, training pairs, thresholds |

## Tests

```bash
# Layer A unit tests (folder must not exist). Test 5 needs 1,000 real case files: set V4_PILOT_DIR to run it.
V4_PILOT_DIR=/path/to/pilot_json python test_ctx.py tests_a

# Layer B unit tests (synthetic data; uses models/model.joblib unless V4_LB_MODEL is set)
python test_consolidate.py tests_b

# full procedure: first folder -> add second folder -> rerun -> all at once (Layer A), then Layer B on top
PYTHON=$(which python) ./v4_test_run.sh /path/to/test_folder /path/to/first_json /path/to/second_json
PYTHON=$(which python) ./v4_consolidate_run.sh /path/to/test_folder
```

`v4_consolidate_run.sh` trains a model on the first-folder state, runs Layer B incrementally and all at once, and compares the two. It then runs `check_consolidated.py`, the real-data checks written for the pilot + TXSD test set, and the unit tests. `score.py` scores a run against the answer key: `python score.py --key /path/to/answer_key --v4dir V4=/path/to/run`. The answer key and the case files are not included.

## Known limitations

- **The LLM review is off by default.** The precision check needs labelled pilot pairs. Those pairs contain party names, so they were removed from `models/model.joblib`. With this model, LLM answers are notes in the review queue and never merge anything.
- **Firms and organisation parties never auto-merge.** On held-out courts the model never reaches 0.98 precision for organisation pairs, because branch offices and corporate-family members look like duplicates. Those pairs only go to the review queue.
- **Repeat officials may still be split.** A judge or lawyer named differently across cases without shared context (no shared case, division or co-occurring entities), or a docket-only name still in the pending pool, can keep two IDs.
- **The surviving ID keeps the oldest name.** The display name can be the shorter form, for example "<first> <middle>" instead of the full name; `name_variants` holds all forms.
- **Some merges are debatable.** Party persons can be joined across linked cases (a complaint and the later indictment) on co-occurrence evidence alone, and some identical counsel names are joined without contact evidence. Every merge is listed in `report.md` with its score and top reasons.
- **Layer B only pairs IDs in the same court**, so a person who appears in several courts keeps one ID per court.

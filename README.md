# V4: entity resolution for court records

V4 reads court case files and gives stable IDs to the **judges, parties, counsel and firms** named in them. It has two layers.

- **Layer A (`v4_run_ctx.py`): fast incremental linking.** One pass per file.
  - **Name gate.** Before a name becomes an entity, the gate decides whether the string is a PERSON, an ORGANISATION or NOT AN ENTITY:
    - two pretrained models vote: spaCy NER on the docket sentence when there is one, and probablepeople;
    - each distinct string is voted on once and cached;
    - when the votes differ, data decides: a known judge in the court, the FJC list, a judge in the case's judge field, a known lawyer, firm or party;
    - then an optional local Ollama model decides;
    - anything still undecided goes to a review file.
  - **Rules the gate enforces.** Judges must be persons. Counsel that is an organisation (an office) stays counsel, flagged `is_organisation` (setting `name_gate.counsel_organisation`; `firm` turns it into a firm mention instead). For parties, the gate decides person vs organisation.
  - **Matching.** Each mention is matched against an index of saved entities, and a merge needs context evidence: same court and a compatible name for judges, shared counsel or co-party for persons, shared contact details for counsel and firms, and identical normalised names for organisations.
  - **Incremental.** Re-running on the same output folder adds only new files (skipped by name + sha256). Old IDs never change.
- **Layer B (`v4_consolidate.py`): learned consolidation.** Runs after Layer A and merges IDs that belong to the same real person or organisation.
  - **The model.** A calibrated gradient-boosting model scores candidate pairs. It was trained on the pilot answer key, the FJC list and automatic name fragments, with court-grouped folds. The features cover names, shared cases and divisions, co-occurring entities and party side. Organisations add relational evidence: shared counsel, the same lawyers using both names, email domains, phones and addresses.
  - **Organisations.** Each pair ends as SAME (merge), RELATED (an office, or a parent / subsidiary; both IDs kept and linked in `related_links.jsonl`) or DIFFERENT.
  - **Thresholds.** A group merges automatically only where held-out precision is >= 0.98. Persons use one threshold per kind; organisations use one per evidence group. The values are in `models/model_card.md`.
  - **Append-only.** Redirects are never undone. The surviving ID is the oldest one.

Shared code:
- `v4_names.py`: the name gate, the one organisation normalisation used by both layers, and display names.
- `v4_readers.py`: input readers chosen by file structure.

Rules and settings:
- Layer A: `v4_rules.yaml`.
- Layer B: `v4_consolidate.yaml`.
- Every list and number in those files has a `source:` note.

## Setup

Python 3.13 (3.12 also works). probablepeople and its `python-crfsuite` dependency install fine on 3.13, so GLiNER2 is not needed as the gate's second vote.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import spacy, probablepeople; spacy.load('en_core_web_md'); print('ok')"
```

- **spaCy model.** `requirements.txt` installs `en_core_web_md` 3.8.0 directly. Alternatively: `python -m spacy download en_core_web_md`.
- **GLiNER2** (document / generic readers). It downloads `fastino/gliner2-base-v1` (about 0.8 GB) on first use.
- **Ollama** (optional `--llm local`). Install from ollama.com, then run `ollama pull qwen2.5:7b`. The model and URL are set in `v4_rules.yaml` (`name_gate.ollama`).
- **Reference data.** `data/` holds the FJC judge list and the Census name files; see `data/README.md`.

## Run

```bash
# first run: --input = .json files, folders of .json files, or .zip files (the reader is chosen by each file's structure)
python v4_run_ctx.py --input /path/to/case_json --output /path/to/run

# add files later to the same output (processed files are skipped; old IDs never change)
python v4_run_ctx.py --input /path/to/new_files --output /path/to/run

# Layer B with the saved model (also for a new court: no retraining); run it again after adding files - only new or
# changed IDs are scored and earlier redirects are kept
python v4_consolidate.py --input /path/to/run --output /path/to/consolidated --model models/model.joblib

# the final result as exactly two files (entities.jsonl, mentions.jsonl)
python v4_consolidate.py --input /path/to/run --output /path/to/consolidated --model models/model.joblib --export /path/to/export

# optional local LLM: settles name-gate disagreements in Layer A (answers cached in the output folder)
python v4_run_ctx.py --input ... --output ... --llm local
# documents read with Ollama instead of GLiNER2
python v4_run_ctx.py --input ... --output ... --doc-model ollama

# train a new Layer B model (needs the answer key folder; without it, FJC + fragments only)
python v4_consolidate.py --input /path/to/run --output /path/to/new_model_run --key /path/to/answer_key
```

**Input readers**, chosen by file structure, not file name:

| Structure | Reader |
|---|---|
| Docket JSON: a `ucid` plus `parties` / `docket` | the field reader |
| OCR document: `filename` + `pages[].blocks[]` with `class`, `confidence`, `bbox`, `text` (e.g. YOLO output) | the document reader |
| Any other JSON | the generic reader: every text field, with its JSON path |

Document and generic readers. All document / generic files of a run are read together, after the dockets, using only models, layout and statistics from the data (no phrase lists, no form-specific rules):

1. **Template learning.** A line repeated (fuzzily, for OCR noise) in documents of two or more cases, or in two or more documents of one case without a name filling it, is form text. Form text never gives a role by itself. Each short template line (a label) is paired, by bbox, with the nearest line holding a name. The label's meaning (party / lawyer / judge / other field) comes from GLiNER2: the extraction model reading the label text, or the zero-shot classifier when the label names several roles.
2. **Participant vs mentioned.** A name that fills its line in a non-prose block (a value line), or that a form label points to, fills a participant slot. Names inside prose go to `mentioned_only.jsonl`, not to entities. The model also has an explicit "only mentioned in the text" class.
3. **One role per name per case.** The role is chosen from the evidence of every document of the case:
   - the paired label's meaning;
   - GLiNER2 name labels (a second pass on the value line alone adds judge / attorney evidence; the model's plaintiff / defendant guesses for person names count half);
   - a bar number / email / phone / address next to the only persons of a block;
   - a law firm in the same block.

   A weak majority (under 70%) goes to `document_review.jsonl`.
4. **Firm vs company.** An organisation is a firm only when attached to a lawyer (same block, or a shared email domain). Otherwise it is a party (when a party slot holds it) or only mentioned.
5. **Docket first.** When docket JSON of the same case is in this run or the output folder, its structured roles override document guesses for the same name.

Every name then passes through the name gate; undecided document names are reviewed, not created. Case ID and court come from the file name and the text, cross-checked. Page, bbox and confidence are kept with each mention; low-confidence names go to `document_review.jsonl`. The settings, each with a source note, are in `v4_rules.yaml` under `documents`. `python doc_report.py <layer A output> --names` prints, per document, the entities by type, the mentioned-only names and the review rows.

**While it runs:**
- Layer A prints `Progress:` / `Reading:` / `Name gate:` / `End step:` lines.
- Layer B prints a line per step.
- `kill -USR1 <pid>` prints the current Python stack (where the run is) to the log without stopping it.
- Every output file is written to a temp file and renamed, so an interrupted save never leaves a half-written file.

**Layer A output files:**

| File | Contents |
|---|---|
| `entities.jsonl` | the entities |
| `mentions.jsonl` | every mention, with its context, role and side |
| `state.json` | ID counters and processed files |
| `uncertain_pairs.jsonl` | close pairs that were not merged |
| `pending_judges.jsonl` | docket judge names not yet promoted |
| `dropped_entities.jsonl` | every drop, with the gate votes where the gate decided |
| `gate_review.jsonl` | names the gate could not decide |
| `name_gate_cache.jsonl` | model votes per distinct string |
| `document_review.jsonl` | document rows to check (low confidence, case-number mismatch, conflicting roles) |
| `mentioned_only.jsonl` | names found only inside document text, not filling a participant slot (written when there are any) |
| `document_model_cache.jsonl` | document model results |

**Layer B output files:**

| File | Contents |
|---|---|
| `entities_consolidated.jsonl` | final entities, with `merged_from`, `related` and party side counts |
| `redirects.jsonl` | old ID -> surviving ID |
| `mentions_resolved.jsonl` | every mention with its final ID |
| `related_links.jsonl` | organisation pairs linked as RELATED |
| `review_queue.jsonl` | pairs left for review |
| `consolidation_state.json` | state for incremental runs |
| `report.md` | merges, RELATED links, organisation names still split, review queue, time |
| `model_card.md` | the model's features, weights, thresholds and training pairs |

**`--export` files:**

| File | Contents |
|---|---|
| `entities.jsonl` | final entities with variants, courts, cases, related IDs; parties carry plaintiff / defendant / other counts; parties and counsel carry `is_organisation` |
| `mentions.jsonl` | every mention with its details, role / side, `is_organisation` and FINAL entity ID; pending docket judge names have a null ID |

## Tests

```bash
# Layer A (49 checks; test 5 needs 1,000 real case files in V4_PILOT_DIR)
V4_PILOT_DIR=/path/to/pilot_json python test_ctx.py tests_a

# Layer B (13 checks, synthetic data, uses models/model.joblib unless V4_LB_MODEL is set)
python test_consolidate.py tests_b

# this release (20 checks): pending-judge scaling (1k / 10k / 50k names), name gate, &/and and office field,
# office -> RELATED, export format, readers by structure, kill -USR1, safe saving, no real names in code or config
V4_PILOT_DIR=/path/to/pilot_json python test_v4_new.py tests_new

# full procedure scripts (folders as arguments)
PYTHON=$(which python) ./v4_test_run.sh /path/to/test_folder /path/to/first_json /path/to/second_json
PYTHON=$(which python) ./v4_consolidate_run.sh /path/to/test_folder
```

- `score.py` scores a run against the answer key: `python score.py --key /path/to/answer_key --v4dir V4=/path/to/run`.
- The answer key and the case files are not included.

## Lists that remain, and why

The gate replaced the keyword lists that decided what is a name: organisation words, docket non-judge keywords, judge non-name words, court-staff phrases for names, the GLiNER2 type stage, and legal-form / abbreviation lists. The removed lists and the reasons are listed in `v4_rules.yaml` under `removed_unused`.

What remains:

- **Field and format structure:**
  - JSON field names (`entity_types`);
  - the docket phrase patterns that locate a judge name;
  - case-number, email, phone, ZIP and bar-number patterns;
  - side by party type / role (`party_roles`).
- **Name cleaning before matching:**
  - title words and title typos ("Judge", "Hon.", "Magistrate");
  - name suffixes (Jr, Sr, III);
  - apostrophes;
  - counsel office tags ("-DOJ");
  - stop-words for organisation initials;
  - one leading article "the" for organisation names.

  These make keys comparable; they never decide whether a string is a name.
- **Matching rules the user set:**
  - the government-party fold;
  - per-district U.S. Attorney / Federal Public Defender offices;
  - Doe placeholders never merge across cases;
  - placeholders ("Attorney at Law");
  - nicknames for first-name compatibility;
  - webmail domains (not evidence).

  The gate did worse than the placeholder list on the pilot ("Attorney at Law" gets split votes), so that list stays.
- **Court-staff roles** (`rawinfo_staff_phrases`): a counsel / firm entry naming a role such as interpreter, probation or clerk is court staff. The gate sees names, not roles.
- **Model label names:**
  - spaCy's own labels mapped to the three outcomes (PERSON; ORG, GPE, NORP, FAC -> ORGANISATION);
  - the GLiNER2 label descriptions.

## Known limitations

- **Document reader (OCR) limits** seen on the 7 test files:
  - Persons in a certificate-of-service list with no form label may be split between counsel and party and go to review: GLiNER2 labels most bare person names plaintiff / defendant.
  - A party name repeated in captions of many cases (e.g. the federal government in criminal judgments) is treated as form text.
  - OCR debris next to a form label can still become a name.

- **The LLM is off by default.** The Layer B `--llm` precision check needs labelled pilot pairs. Those contain party names and were removed from `models/model.joblib`, so Layer B LLM answers are notes only.
- **Firms and organisation parties never auto-merge in Layer B.** On the pilot answer key, no organisation evidence group reached 0.98 held-out precision: the key has few organisation positives, and it splits spelling variants. Such pairs become RELATED links or review items.
- **Organisation names are normalised with the parser's legal-type parts only.** "Corp." vs "Corporation" (which the parser tags as an organisation word, not a legal form) stay separate IDs. A setting that also removes one final organisation word exists (`organisations.legal_form_rule`); it raises party recall on the pilot but lowers precision.
- **Counsel that are offices** (e.g. "Federal Defenders") stay counsel mentions flagged `is_organisation` (default setting `counsel`). With `name_gate.counsel_organisation: firm` they become firm mentions instead; the pilot answer key has no firms from the counsel field, so that lowers firm extraction precision there (0.9813 -> 0.9620).
- **Repeat officials may still be split.** A judge or lawyer named differently with no shared context, or a docket-only name still in the pending pool, can keep two IDs.
- **The surviving ID is the oldest one**, but `canonical_name` is rebuilt from the most complete variant.
- **Some merges are debatable.** Party persons can be joined across linked cases on co-occurrence evidence. Every merge is listed in `report.md` with its score and top reasons.
- **The name gate is not perfect.**
  - Unusual names can get split votes and end in `gate_review.jsonl` (kept for docket fields, not created for documents).
  - An all-caps title-laden string can still be read as an organisation.
  - Both cases are logged with the two votes.

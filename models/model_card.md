# Layer B model card

- Model kept: **hist_gradient_boosting** (isotonic-calibrated; the better held-out average precision of the two)
- Trained on the V4 output folder `run_after_pilot`; version `layer-b-1`
- Candidate filter (speed only): same type AND (share a case OR same court with char n-gram TF-IDF cosine >= 0.4384). The cut is the 2th percentile of the cosine over labelled same-court positive pairs. The filter keeps 62 of 62 labelled real same-court positive pairs.

## Thresholds (from held-out court folds)

- **T_high, per pair kind** (merge if p >= T_high): the smallest score at which held-out precision stays >= 0.98 from the top down, needing >= 20 positives of that kind, and at which the real labelled pairs of that kind (answer key, FJC) also keep that precision (no real negative above it when there are no real positives). A kind that never reaches it gets no automatic merges (review queue only): counsel 0.7884, firm not reached, judge 0.2363, party_org not reached, party_person 0.8710, party_person_org not reached.
- **T_low = 0.7851**: review queue for T_low <= p < T_high. 95% of held-out true pairs score at or above it.

Held-out results at T_high (by kind and by label source):

| group | pairs | positives | pairs >= T_high | precision | recall |
|---|---|---|---|---|---|
| all kinds | 268860 | 7183 | 6804 | 0.9850 | 0.9330 |
| counsel | 50931 | 2940 | 2807 | 0.9804 | 0.9361 |
| firm | 3011 | 32 | 0 |  | 0.0000 |
| judge | 4430 | 1267 | 1284 | 0.9805 | 0.9937 |
| party_org | 135726 | 26 | 0 |  | 0.0000 |
| party_person | 20617 | 2917 | 2713 | 0.9919 | 0.9225 |
| party_person_org | 54145 | 1 | 0 |  | 0.0000 |
| firm, real labels only | 3011 | 32 | 0 |  | 0.0000 |
| judge, real labels only | 263 | 1 | 1 | 1.0000 | 1.0000 |
| party_org, real labels only | 135726 | 26 | 0 |  | 0.0000 |
| party_person, real labels only | 346 | 2 | 0 |  | 0.0000 |
| party_person_org, real labels only | 7185 | 1 | 0 |  | 0.0000 |
| source: answer_key | 146399 | 61 | 0 |  | 0.0000 |
| source: fjc | 132 | 1 | 1 | 1.0000 | 1.0000 |
| source: fragment | 122329 | 7121 | 6803 | 0.9850 | 0.9410 |

## Both models on held-out courts

| model | average precision | Brier | log loss | AP by source |
|---|---|---|---|---|
| logistic_regression | 0.9911 | 0.0011 | 0.0051 | answer_key 0.528, fjc 1.000, fragment 0.994 |
| hist_gradient_boosting | 0.9965 | 0.0009 | 0.0035 | answer_key 0.439, fjc 1.000, fragment 0.998 |

## Training pairs by source (label 1 = same, 0 = different)

| source | type | label 1 | label 0 |
|---|---|---|---|
| answer_key | firm | 32 | 2979 |
| answer_key | judge | 0 | 131 |
| answer_key | party | 29 | 143228 |
| fjc | judge | 1 | 131 |
| fragment | counsel | 2940 | 47991 |
| fragment | judge | 1266 | 2901 |
| fragment | party | 2915 | 64316 |

- Answer-key IDs mapped to one gold entity: 2097 (IDs matching 2+ gold entities, skipped: 7)
- Judge IDs mapped to exactly one FJC judge: 451
- Synthetic fragments: 7240; candidate pairs in the training run: 416055
- Split: group k-fold by court; a test court is never seen in training (outer folds for the held-out numbers, inner folds for calibration).

## Features, logistic-regression weights (standardised) and gradient-boosting permutation importance

| feature | LR weight | HGB importance (AP drop) |
|---|---|---|
| cos | +0.417 | 0.0010 |
| jw | -0.239 | 0.0005 |
| prefix | +0.536 | 0.0001 |
| subset | +2.213 | 0.0913 |
| first_compat | +1.625 | 0.0053 |
| surname_match | +1.080 | 0.0000 |
| word_to_initial | +0.153 | 0.0000 |
| suffix_conflict | -0.684 | 0.0027 |
| one_word | -0.934 | 0.0000 |
| both_one_word | +0.811 | 0.0000 |
| ntok_short | +0.517 | 0.0000 |
| ntok_long | -0.449 | 0.0001 |
| case_jaccard | +0.568 | 0.0061 |
| shared_cases | +0.945 | 0.0001 |
| same_court | +0.000 | 0.0000 |
| same_division | +0.975 | 0.0000 |
| co_jaccard | +1.189 | 0.0007 |
| co_shared | -0.476 | 0.0010 |
| date_overlap | +0.301 | 0.0000 |
| date_gap | -0.107 | 0.0000 |
| title_compat | +0.253 | -0.0000 |
| side_agree | +2.915 | 0.0026 |
| count_ratio | -0.100 | 0.0001 |
| count_min_log | -0.552 | 0.0000 |
| comp_court | -0.424 | 0.0002 |
| comp_div | -0.394 | 0.0373 |
| is_judge | +0.978 | 0.0001 |
| is_party | -0.503 | 0.0001 |
| is_counsel | +0.005 | 0.0000 |
| is_firm | +0.770 | 0.0000 |
| is_person | +1.727 | 0.0138 |

Feature meanings are in the FEATURES list at the top of v4_consolidate.py.

## Fixed constraints (the only hand rules)

1. never mix entity types; 2. never put two different FJC judges in one ID; 3. never put two different full first names with the same surname in one ID. Merges are applied in descending score and also respect the model's own keep-apart scores (p < T_low) between the two groups, so no merge chains through a pair the model separated.

## Pilot answer key, court-held-out (score.py, mentions mapped through redirects)

| type | before P | before R | before F1 | after P | after R | after F1 |
|---|---|---|---|---|---|---|
| judge | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| party | 0.9881 | 0.9516 | 0.9695 | 0.9881 | 0.9516 | 0.9695 |
| firm | 1.0000 | 0.8062 | 0.8927 | 1.0000 | 0.8062 | 0.8927 |

Merges by type in that run: {'judge': 8, 'party': 5, 'counsel': 6}; review queue by type: {'firm': 96, 'judge': 2, 'party': 16}


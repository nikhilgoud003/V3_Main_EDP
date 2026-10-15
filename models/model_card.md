# Layer B model card

- Model kept: **hist_gradient_boosting** (isotonic-calibrated; the better held-out average precision of the two)
- Trained on the pilot (1,000 files); version `layer-b-2`
- Candidate filter (speed only): same type AND (share a case OR same court with char n-gram TF-IDF cosine >= 0.4299). The cut is the 2th percentile of the cosine over labelled same-court positive pairs. The filter keeps 120 of 120 labelled real same-court positive pairs.

## Thresholds (from held-out court folds)

- **T_high, per pair kind** (merge if p >= T_high): the smallest score at which held-out precision stays >= 0.98 from the top down, needing >= 20 positives of that kind, and at which the real labelled pairs of that kind (answer key, FJC) also keep that precision (no real negative above it when there are no real positives). A kind that never reaches it gets no automatic merges (review queue only): counsel 0.7535, firm|contact not reached, firm|counsel not reached, firm|name not reached, judge 0.3287, party_org|contact not reached, party_org|counsel not reached, party_org|name not reached, party_person 0.8315, party_person_org not reached.
- **T_low = 0.6749**: review queue for T_low <= p < T_high. 95% of held-out true pairs score at or above it.

Held-out results at T_high (by kind and by label source):

| group | pairs | positives | pairs >= T_high | precision | recall |
|---|---|---|---|---|---|
| all kinds | 304903 | 7286 | 6610 | 0.9834 | 0.8921 |
| counsel | 52613 | 2937 | 2784 | 0.9802 | 0.9292 |
| firm|contact | 78 | 22 | 0 |  | 0.0000 |
| firm|counsel | 2 | 1 | 0 |  | 0.0000 |
| firm|name | 2871 | 2 | 0 |  | 0.0000 |
| judge | 4305 | 1255 | 1273 | 0.9804 | 0.9944 |
| party_org|contact | 1581 | 1 | 0 |  | 0.0000 |
| party_org|counsel | 2522 | 19 | 0 |  | 0.0000 |
| party_org|name | 119650 | 60 | 0 |  | 0.0000 |
| party_person | 28445 | 2983 | 2553 | 0.9882 | 0.8458 |
| party_person_org | 92836 | 6 | 0 |  | 0.0000 |
| firm|contact, real labels only | 78 | 22 | 0 |  | 0.0000 |
| firm|counsel, real labels only | 2 | 1 | 0 |  | 0.0000 |
| firm|name, real labels only | 2871 | 2 | 0 |  | 0.0000 |
| judge, real labels only | 261 | 1 | 1 | 1.0000 | 1.0000 |
| party_org|contact, real labels only | 1581 | 1 | 0 |  | 0.0000 |
| party_org|counsel, real labels only | 2522 | 19 | 0 |  | 0.0000 |
| party_org|name, real labels only | 119650 | 60 | 0 |  | 0.0000 |
| party_person, real labels only | 1307 | 8 | 0 |  | 0.0000 |
| party_person_org, real labels only | 21468 | 6 | 0 |  | 0.0000 |
| source: answer_key | 149609 | 119 | 0 |  | 0.0000 |
| source: fjc | 131 | 1 | 1 | 1.0000 | 1.0000 |
| source: fragment | 155163 | 7166 | 6609 | 0.9834 | 0.9069 |

## Both models on held-out courts

| model | average precision | Brier | log loss | AP by source |
|---|---|---|---|---|
| logistic_regression | 0.9820 | 0.0015 | 0.0067 | answer_key 0.464, fjc 1.000, fragment 0.988 |
| hist_gradient_boosting | 0.9893 | 0.0013 | 0.0054 | answer_key 0.131, fjc 1.000, fragment 0.996 |

## Training pairs by source (label 1 = same, 0 = different)

| source | type | label 1 | label 0 |
|---|---|---|---|
| answer_key | firm | 25 | 2926 |
| answer_key | judge | 0 | 130 |
| answer_key | party | 94 | 146434 |
| fjc | judge | 1 | 130 |
| fragment | counsel | 2937 | 49676 |
| fragment | judge | 1254 | 2790 |
| fragment | party | 2975 | 95531 |

- Answer-key IDs mapped to one gold entity: 2135 (IDs matching 2+ gold entities, skipped: 11)
- Judge IDs mapped to exactly one FJC judge: 451
- Synthetic fragments: 7290; candidate pairs in the training run: 451656
- Split: group k-fold by court; a test court is never seen in training (outer folds for the held-out numbers, inner folds for calibration).

## Features, logistic-regression weights (standardised) and gradient-boosting permutation importance

| feature | LR weight | HGB importance (AP drop) |
|---|---|---|
| cos | +0.936 | 0.0009 |
| jw | -0.141 | 0.0004 |
| prefix | -0.129 | 0.0000 |
| subset | +1.717 | 0.1849 |
| first_compat | +1.693 | 0.0028 |
| surname_match | +0.106 | 0.0000 |
| word_to_initial | -0.019 | 0.0000 |
| suffix_conflict | -0.669 | 0.0053 |
| one_word | -0.687 | 0.0000 |
| both_one_word | +0.322 | 0.0000 |
| ntok_short | +0.022 | -0.0000 |
| ntok_long | -0.323 | 0.0001 |
| case_jaccard | +0.901 | 0.0151 |
| shared_cases | +0.845 | 0.0001 |
| same_court | +0.000 | 0.0000 |
| same_division | +0.435 | 0.0000 |
| co_jaccard | +0.645 | 0.0003 |
| co_shared | -0.298 | 0.0010 |
| date_overlap | +0.231 | -0.0000 |
| date_gap | -0.099 | 0.0000 |
| title_compat | +0.188 | 0.0000 |
| side_agree | +1.974 | 0.0021 |
| count_ratio | -0.101 | 0.0001 |
| count_min_log | +0.043 | 0.0001 |
| comp_court | -0.504 | 0.0000 |
| comp_div | -0.312 | 0.0584 |
| is_judge | +0.739 | 0.0000 |
| is_party | -0.364 | -0.0000 |
| is_counsel | +0.058 | 0.0000 |
| is_firm | +0.362 | -0.0000 |
| is_person | +1.314 | 0.0227 |
| org_pair | -0.212 | 0.0000 |
| org_same_core | -0.057 | 0.0000 |
| org_same_loose | +0.192 | 0.0000 |
| org_office_differs | -0.036 | 0.0000 |
| org_core_prefix | -0.080 | 0.0000 |
| org_shared_counsel | -0.265 | 0.0000 |
| org_shared_domain | +0.748 | -0.0000 |
| org_shared_phone | -0.400 | -0.0000 |
| org_shared_address | +0.128 | -0.0000 |

Feature meanings are in the FEATURES list at the top of v3_consolidate.py.

## Fixed constraints (the only hand rules)

1. never mix entity types; 2. never put two different FJC judges in one ID; 3. never put two different full first names with the same surname in one ID; 4. never merge organisations listed as separate parties in one case. Merges are applied in descending score and also respect the model's own keep-apart scores (p < T_low) between the two groups, so no merge chains through a pair the model separated. Organisation thresholds are set per evidence group (shared counsel / shared contact details / name only); a group that never reaches the precision target merges nothing automatically. Organisation pairs that stay apart are linked as RELATED in related_links.jsonl: an office (same core name, different office) or a parent / subsidiary (one core name is the first words of the other, with relational evidence or a score in the review band).

## Pilot answer key, court-held-out (score.py, mentions mapped through redirects)

| type | before P | before R | before F1 | after P | after R | after F1 |
|---|---|---|---|---|---|---|
| judge | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| party | 0.9838 | 0.8743 | 0.9258 | 0.9838 | 0.8743 | 0.9258 |
| firm | 0.9962 | 0.8106 | 0.8938 | 0.9962 | 0.8106 | 0.8938 |

Merges by type in that run: {'judge': 8, 'party': 22, 'counsel': 6}; review queue by type: {'firm': 76, 'party': 47, 'judge': 1}


# Reference data (public)

| File | Used by | Source |
|---|---|---|
| `judges_fjc.csv` | Layer A (docket judges: FJC-listed names are promoted from the pending pool, option A) and Layer B (FJC label source; the "two different FJC judges" safety check) | Federal Judicial Center, *Biographical Directory of Article III Federal Judges, 1789-present*, CSV export from fjc.gov (History of the Federal Judiciary, "Judges" export). Public domain (US government work). |
| `census1990_dist.male.first.txt` | Layer A (first-name-like one-word judge names) and Layer B (a party ID is a person when its first word is a first name) | US Census Bureau, 1990 Census name files, `dist.male.first`, www2.census.gov/topics/genealogy/1990surnames/. Public domain. |
| `census1990_dist.female.first.txt` | same as above | same source, `dist.female.first` |
| `census1990_dist.all.last.txt` | Layer A (a word is first-name-like when it is more frequent as a first name than as a surname) | same source, `dist.all.last` |

The files are unchanged copies of the downloads. Paths are set in `v3_rules.yaml` (Layer A) and `v3_consolidate.yaml` (Layer B), relative to those files.

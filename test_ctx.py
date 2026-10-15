#!/usr/bin/env python3
"""Tests for v3_run_ctx.py. Writes only under ./tests_ctx/ (must not exist yet).

  1. "J Smith" judge and "John Smith" judge in the same court -> one ID
  2. "J Smith" party vs "John Smith" judge -> two IDs
  3. "John Smith" party in two unrelated cases -> two IDs
  4. the same with a shared firm -> one ID
  5. incremental 800 + 200 on the pilot keeps 100% of old IDs, and re-adding a file changes nothing
"""
import collections
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "v3_run_ctx.py"
# Test 5 needs a folder of 1,000 real PACER case files; set V3_PILOT_DIR to run it (skipped otherwise).
PILOT = Path(os.environ["V3_PILOT_DIR"]).expanduser() if os.environ.get("V3_PILOT_DIR") else None
T = HERE / (sys.argv[1] if len(sys.argv) > 1 else "tests_ctx")  # test folder (must not exist)
results = []


def case(ucid, court, judge=None, parties=(), referred=()):
    return {"ucid": ucid, "court": court, "judge": judge, "referred_judges": list(referred), "docket": [],
            "parties": [{"name": n, "counsel": [{"name": cn, "entity_info": info} for cn, info in counsel]}
                        for n, counsel in parties]}


def write_cases(folder, cases):
    folder.mkdir(parents=True)
    for c in cases:
        (folder / (c["ucid"].replace(";;", "-").replace(":", "-") + ".json")).write_text(json.dumps(c))


def run(inputs, out, *extra):
    r = subprocess.run([sys.executable, str(SCRIPT), "--input", *map(str, inputs), "--output", str(out), *extra],
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stdout, r.stderr)
        raise SystemExit("run failed")
    return r.stdout


def load(out):
    ents = [json.loads(l) for l in open(out / "entities.jsonl")]
    ments = {m["mention_id"]: m for m in map(json.loads, open(out / "mentions.jsonl"))}
    return ents, ments


def ids_for(ments, etype, raw):
    return sorted({m["entity_id"] for m in ments.values() if m["entity_type"] == etype and m["raw_name"] == raw})


def check(name, ok, detail):
    ok = bool(ok)
    results.append((name, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  ({detail})")


def info(office, addr, phone, email):
    return {"office_name": office, "address": addr, "phone": phone, "email": email}


if T.exists():
    raise SystemExit(f"{T} already exists; remove it or choose another folder")
T.mkdir()

# --- tests 1 and 2: judges, and party vs judge ---
write_cases(T / "in_judges", [
    case("nyed;;1:16-cv-00001", "nyed", judge="John Smith",
         parties=[("Acme Widgets Inc", [("Ann Lee", info("Lee LLP", "1 Main St, Brooklyn, NY 11201", "718-555-0101", "alee@lee.example.com"))])]),
    case("nyed;;1:16-cv-00002", "nyed", judge="J. Smith",
         parties=[("Beta Tools LLC", [("Bob Kim", info("Kim PC", "9 Elm St, Brooklyn, NY 11201", "718-555-0202", "bkim@kim.example.com"))])]),
    case("nyed;;1:16-cv-00003", "nyed", judge="John Smith",
         parties=[("J. Smith", [("Cara Diaz", info("Diaz Law", "5 Oak St, Queens, NY 11375", "718-555-0303", "cdiaz@diaz.example.com"))])]),
])
out = T / "out_judges"
run([T / "in_judges"], out)
_, m = load(out)
j_full, j_init = ids_for(m, "judge", "John Smith"), ids_for(m, "judge", "J. Smith")
check("1. J Smith judge + John Smith judge, same court -> one ID", len(j_full) == 1 and j_full == j_init, f"{j_full} / {j_init}")
p_js = ids_for(m, "party", "J. Smith")
check("2. J Smith party vs John Smith judge -> two IDs", len(p_js) == 1 and p_js != j_full, f"party {p_js} vs judge {j_full}")

# --- test 3: same party name, unrelated cases ---
write_cases(T / "in_parties_unrelated", [
    case("ilnd;;1:16-cv-00010", "ilnd", parties=[
        ("John Smith", [("Dana White", info("White Law Group", "100 W Madison St, Chicago, IL 60602", "312-555-1000", "dwhite@white.example.com"))]),
        ("Gamma Corp", [("Eli Stone", info("Stone LLP", "200 N State St, Chicago, IL 60601", "312-555-2000", "estone@stone.example.com"))])]),
    case("ilnd;;1:17-cv-00020", "ilnd", parties=[
        ("John Smith", [("Fay Gold", info("Gold & Partners", "300 S Wacker Dr, Chicago, IL 60606", "312-555-3000", "fgold@gold.example.com"))]),
        ("Delta Inc", [("Gus Hill", info("Hill PC", "400 E Ohio St, Chicago, IL 60611", "312-555-4000", "ghill@hill.example.com"))])]),
])
out = T / "out_parties_unrelated"
run([T / "in_parties_unrelated"], out)
_, m = load(out)
p3 = ids_for(m, "party", "John Smith")
check("3. John Smith party in two unrelated cases -> two IDs", len(p3) == 2, f"{p3}")

# --- test 4: same party name, shared firm ---
acme = info("Acme Law LLP", "50 Park Ave, Suite 10, New York, NY 10016", "212-555-5000", "")
write_cases(T / "in_parties_shared_firm", [
    case("nysd;;1:16-cv-00030", "nysd", parties=[
        ("John Smith", [("Hal Ives", acme)]), ("Epsilon LLC", [("Ida Jones", info("Jones PC", "60 Wall St, New York, NY 10005", "212-555-6000", "ijones@jones.example.com"))])]),
    case("nysd;;1:17-cv-00040", "nysd", parties=[
        ("John Smith", [("Jack Kerr", acme)]), ("Zeta Inc", [("Kim Long", info("Long LLP", "70 Pine St, New York, NY 10005", "212-555-7000", "klong@long.example.com"))])]),
])
out = T / "out_parties_shared_firm"
run([T / "in_parties_shared_firm"], out)
_, m = load(out)
p4, f4 = ids_for(m, "party", "John Smith"), ids_for(m, "firm", "Acme Law LLP")
check("4. John Smith party, two cases, shared firm -> one ID", len(p4) == 1, f"party {p4}, firm {f4}")

# --- test 6 (chaining guard): J Smith must not link John Smith and James Smith ---
write_cases(T / "in_chain", [
    case("nyed;;1:18-cv-00011", "nyed", judge="John Smith"),
    case("nyed;;1:18-cv-00012", "nyed", judge="J. Smith"),
    case("nyed;;1:18-cv-00013", "nyed", judge="James Smith"),
])
for order in ("name", "reverse"):
    out = T / f"out_chain_{order}"
    run([T / "in_chain"], out, "--order", order)
    _, m = load(out)
    jo, js, ja = ids_for(m, "judge", "John Smith"), ids_for(m, "judge", "J. Smith"), ids_for(m, "judge", "James Smith")
    # J. Smith joins whichever full name is saved first (John in name order, James in reverse order),
    # but John and James must never end up in one entity.
    check(f"6. chaining guard ({order} order): John and James never share an ID; J. Smith joins exactly one",
          len(jo) == 1 and len(ja) == 1 and ja != jo and len(js) == 1 and js[0] in (jo[0], ja[0]),
          f"John {jo}, J. {js}, James {ja}")

# --- test 7 (surname-only judges) ---
write_cases(T / "in_surname_only", [
    case("ilnd;;1:09-cv-00001", "ilnd", judge="Oren Castile", referred=["Judge Castile"]),
    case("ilnd;;1:09-cv-00002", "ilnd", judge="Maria Castile"),
    case("ilnd;;1:09-cv-00003", "ilnd", judge="Judge Castile"),
])
out = T / "out_surname_only"
run([T / "in_surname_only"], out)
_, m = load(out)
rc, mc_ = ids_for(m, "judge", "Oren Castile"), ids_for(m, "judge", "Maria Castile")
jc = {mm["entity_id"] for mm in m.values() if mm["entity_type"] == "judge" and mm["raw_name"] == "Judge Castile"}
jc_case1 = {mm["entity_id"] for mm in m.values() if mm["raw_name"] == "Judge Castile" and mm["ucid"] == "ilnd;;1:09-cv-00001"}
jc_case3 = {mm["entity_id"] for mm in m.values() if mm["raw_name"] == "Judge Castile" and mm["ucid"] == "ilnd;;1:09-cv-00003"}
unc = [json.loads(l) for l in open(out / "uncertain_pairs.jsonl")]
check("7a. 'Judge Castile' in the same case as Oren Castile -> Oren's ID", jc_case1 == set(rc), f"{jc_case1} vs {rc}")
check("7b. 'Judge Castile' alone, two Castile judges in the court -> neither, logged uncertain",
      jc_case3 and not (jc_case3 & set(rc + mc_)) and any("surname-only" in u["reason"] for u in unc),
      f"case3 {jc_case3}, Oren {rc}, Maria {mc_}, uncertain rows {len(unc)}")

# --- test 8 (counsel tags): "-DOJ" is not the surname ---
doj = info("U.S. Department of Justice", "950 Pennsylvania Ave NW, Washington, DC 20530", "202-555-0000", "")
write_cases(T / "in_counsel_tags", [
    case("txnd;;2:17-cr-00017", "txnd", parties=[("Alpha Co", [("Jared R Frost-DOJ", {**doj, "email": "jared.frost@example.gov"}),
                                                             ("Joel Kent Dale-DOJ", {**doj, "email": "joel.dale@example.gov"})])]),
    case("txnd;;4:16-cv-00102", "txnd", parties=[("Beta Co", [("J Neil Oster-DOJ", {**doj, "email": "j.oster@example.gov"}),
                                                             ("Jared R Frost-DOJ", {**doj, "email": "jared.frost@example.gov"})])]),
])
out = T / "out_counsel_tags"
run([T / "in_counsel_tags"], out)
_, m = load(out)
h, f_, w = ids_for(m, "counsel", "Jared R Frost-DOJ"), ids_for(m, "counsel", "Joel Kent Dale-DOJ"), ids_for(m, "counsel", "J Neil Oster-DOJ")
check("8. counsel '-DOJ' tag: Frost (2 cases, same email) one ID; Frost, Dale, Oster three IDs",
      len(h) == 1 and len({*h, *f_, *w}) == 3, f"Frost {h}, Dale {f_}, Oster {w}")

# --- test 9 (step 1): behaviour comes from v3_rules.yaml ---
import yaml
write_cases(T / "in_rules", [
    case("txsd;;1:16-cr-00001", "txsd", parties=[("USA", [("Ann Lee", info("Lee LLP", "1 Main St, Houston, TX 77002", "713-555-0101", "alee@lee.example.com"))])]),
    case("txsd;;1:16-cr-00002", "txsd", parties=[("USA", [("Bob Kim", info("Kim PC", "9 Elm St, Houston, TX 77002", "713-555-0202", "bkim@kim.example.com"))])]),
])
rules = yaml.safe_load(open(HERE / "v3_rules.yaml"))
rules["government_party_names"]["names"] = [n for n in rules["government_party_names"]["names"] if n != "usa"]
custom = T / "rules_without_usa.yaml"
custom.write_text(yaml.safe_dump(rules, sort_keys=False, allow_unicode=True))
run([T / "in_rules"], T / "out_rules_default")
run([T / "in_rules"], T / "out_rules_custom", "--rules", str(custom))
_, md = load(T / "out_rules_default")
_, mc = load(T / "out_rules_custom")
check("9. rules file drives behaviour: 'USA' folds with default rules, not when 'usa' is removed from the YAML",
      len(ids_for(md, "party", "USA")) == 1 and len(ids_for(mc, "party", "USA")) == 2,
      f"default {ids_for(md, 'party', 'USA')}, custom {ids_for(mc, 'party', 'USA')}")

# --- test 10 (step 2): judges ---
cases10 = [case(f"txsd;;7:16-cv-{i:05d}", "txsd", judge="Rory Crandle") for i in range(1, 4)]
cases10 += [case("txsd;;7:16-cv-00010", "txsd", judge="Rry Crandle")]                               # typo -> Randy
cases10 += [case(f"txsd;;4:16-cv-{i:05d}", "txsd", judge="Gray Hollis") for i in range(1, 21)]    # 20+ each:
cases10 += [case(f"txsd;;4:17-cv-{i:05d}", "txsd", judge="Gary Hollis") for i in range(1, 21)]    # not merged
cases10 += [case("txsd;;1:16-cv-00001", "txsd", judge="Ana Lopez"),
            case("txsd;;2:16-cv-00001", "txsd", judge="Ben Lopez"),
            case("txsd;;1:17-cv-00002", "txsd", judge="Judge Lopez"),                              # division 1 -> Ana
            case("txsd;;3:16-cv-00001", "txsd", judge="Magistrate Judge Cora Diaz"),
            case("txsd;;3:16-cv-00002", "txsd", judge="District Judge Dan Diaz"),
            case("txsd;;3:17-cv-00003", "txsd", judge="Magistrate Judge Diaz"),                    # title -> Cora
            case("txsd;;5:16-cv-00001", "txsd", judge="Eve Ruiz"),
            case("txsd;;5:16-cv-00002", "txsd", judge="Finn Ruiz"),
            case("txsd;;5:17-cv-00003", "txsd", judge="Judge Ruiz"),                               # ambiguous -> unresolved
            case("txsd;;5:17-cv-00004", "txsd", judge="Judge Ruiz"),
            case("txsd;;6:16-cv-00001", "txsd", judge="Judge Unassigned", referred=["Jury Selection"])]
write_cases(T / "in_judges2", cases10)
out = T / "out_judges2"
run([T / "in_judges2"], out)
ents10, m = load(out)
eid = lambda raw, ucid=None: {mm["entity_id"] for mm in m.values() if mm["entity_type"] == "judge" and mm["raw_name"] == raw
                              and (ucid is None or mm["ucid"] == ucid)}
check("10a. typo: 'Rry Crandle' joins 'Rory Crandle' (same court, same surname, 1 edit)", eid("Rry Crandle") == eid("Rory Crandle") and len(eid("Rory Crandle")) == 1,
      f"{eid('Rry Crandle')} vs {eid('Rory Crandle')}")
check("10b. typo blocked: 'Gary Hollis' and 'Gray Hollis' both have 20+ mentions -> two IDs", eid("Gary Hollis") != eid("Gray Hollis"),
      f"{eid('Gary Hollis')} vs {eid('Gray Hollis')}")
check("10c. surname-only: 'Judge Lopez' in division 1 -> Ana Lopez (division 1), not Ben Lopez (division 2)", eid("Judge Lopez") == eid("Ana Lopez"),
      f"{eid('Judge Lopez')} vs Ana {eid('Ana Lopez')}, Ben {eid('Ben Lopez')}")
check("10d. title as evidence: 'Magistrate Judge Diaz' -> the magistrate Cora Diaz", eid("Magistrate Judge Diaz") == eid("Magistrate Judge Cora Diaz"),
      f"{eid('Magistrate Judge Diaz')} vs Cora {eid('Magistrate Judge Cora Diaz')}, Dan {eid('District Judge Dan Diaz')}")
unres = [e for e in ents10 if e.get("unresolved")]
ruiz = eid("Judge Ruiz")
check("10e. ambiguous surname-only -> one unresolved record per surname per court, flagged, no sjid",
      len(ruiz) == 1 and len(unres) == 1 and unres[0]["entity_id"] in ruiz and unres[0]["sjid"] is None
      and all(mm.get("unresolved") for mm in m.values() if mm["raw_name"] == "Judge Ruiz"),
      f"Judge Ruiz -> {ruiz}; unresolved records {[e['entity_id'] for e in unres]}")
check("10f. non-name judge entries dropped ('Judge Unassigned', 'Jury Selection')", not eid("Judge Unassigned") and not eid("Jury Selection"),
      f"{eid('Judge Unassigned')} {eid('Jury Selection')}")

# --- test 11 (step 4): agencies ---
write_cases(T / "in_agencies", [
    case("txsd;;4:16-cv-00001", "txsd", parties=[("Northern Department of Public Safety", [])]),
    case("txsd;;4:16-cv-00002", "txsd", parties=[("NDPS", [])]),
    case("txsd;;4:16-cv-00003", "txsd", parties=[("Northern Dept. of Public Safety - Field Operations Division", [])]),
    case("txsd;;4:16-cv-00004", "txsd", parties=[("NDPS-FOD", [])]),
    case("txsd;;4:16-cv-00005", "txsd", parties=[("Western State Medical Branch", [])]),
    case("txsd;;4:16-cv-00006", "txsd", parties=[("WSMB", [])]),
    case("nysd;;1:16-cv-00007", "nysd", parties=[("NDPS", [])]),
    case("txsd;;4:16-cv-00008", "txsd", parties=[("Big-Box Stores Texas, L.L.C.", [])]),
    case("txsd;;4:16-cv-00009", "txsd", parties=[("Big Box Stores Texas LLC", [])]),
    case("txsd;;4:16-cv-00010", "txsd", parties=[("Brown & Green Corp", [])]),
    case("txsd;;4:16-cv-00011", "txsd", parties=[("Brown and Green Corporation", [])]),
    case("txsd;;4:16-cv-00012", "txsd", parties=[("Brown and Green Corp", [])]),
    case("txsd;;4:16-cv-00013", "txsd", parties=[("Brown + Green, Corp.", [])]),
    case("txsd;;4:16-cv-00014", "txsd", parties=[("Northern Department of Public Safety - Field Operations Division", [])]),
])
out = T / "out_agencies"
run([T / "in_agencies"], out)
_, m = load(out)
pid = lambda raw, ucid=None: {mm["entity_id"] for mm in m.values() if mm["entity_type"] == "party" and mm["raw_name"] == raw
                              and (ucid is None or mm["ucid"] == ucid)}
ndps = pid("Northern Department of Public Safety")
# finish job (fix 3): text after a final " - " is the office, so "<org> - <unit>" is its own ID with that office
# (Layer B links it to the parent as RELATED / office); the abbreviation list ("Dept." = "Department") and the
# division-word list were replaced by the parser-based normalisation, and a glued "-FOD" is part of the name
fod = "Northern Department of Public Safety - Field Operations Division"
fod_office = {(mm.get("context") or {}).get("office", [""])[0] for mm in m.values() if mm["raw_name"] == fod}
check("11a. NDPS full name and 'NDPS' (same court) -> one ID; '<org> - <unit>' keeps the unit as its office, own ID",
      len(ndps) == 1 and pid("NDPS", "txsd;;4:16-cv-00002") == ndps and pid(fod) and not (pid(fod) & ndps)
      and fod_office == {"field operations division"},
      f"{ndps} {pid('NDPS', 'txsd;;4:16-cv-00002')} office-variant {pid(fod)} office {fod_office}")
check("11b. 'WSMB' = initials of 'Western State Medical Branch' in the same court -> one ID",
      pid("WSMB") == pid("Western State Medical Branch"), f"{pid('WSMB')} vs {pid('Western State Medical Branch')}")
check("11c. 'NDPS' in another court is not merged (acronyms are court-scoped)", not (pid("NDPS", "nysd;;1:16-cv-00007") & ndps),
      f"nysd {pid('NDPS', 'nysd;;1:16-cv-00007')} vs txsd {ndps}")
check("11d. punctuation / dotted legal form: 'Big-Box Stores Texas, L.L.C.' = 'Big Box Stores Texas LLC'",
      pid("Big-Box Stores Texas, L.L.C.") == pid("Big Box Stores Texas LLC"), "")
# finish job (fix 3): "&", "and", "+", commas and periods are treated alike ("Corp" vs "Corporation" is not a legal-form
# difference to the parser, so that pair is no longer part of this check)
check("11e. '&' vs 'and' vs '+': 'Brown & Green Corp' = 'Brown and Green Corp' = 'Brown + Green, Corp.'",
      pid("Brown & Green Corp") and pid("Brown & Green Corp") == pid("Brown and Green Corp") == pid("Brown + Green, Corp."),
      f"{pid('Brown & Green Corp')} {pid('Brown and Green Corp')} {pid('Brown + Green, Corp.')}")

# --- test 11f (step 4): both arrival orders ---
out = T / "out_agencies_reverse"
run([T / "in_agencies"], out, "--order", "reverse")
_, m = load(out)
ndps_r = pid("Northern Department of Public Safety")
check("11f. reverse order (acronyms arrive first): same NDPS and WSMB groups",
      len(ndps_r) == 1 and pid("NDPS", "txsd;;4:16-cv-00002") == ndps_r and not (pid(fod) & ndps_r)
      and pid("WSMB") == pid("Western State Medical Branch") and not (pid("NDPS", "nysd;;1:16-cv-00007") & ndps_r),
      f"NDPS {ndps_r} / {pid('NDPS', 'txsd;;4:16-cv-00002')}; WSMB {pid('WSMB')} vs {pid('Western State Medical Branch')}")

# --- test 12 (step 4): a lone all-caps word stays a person ---
write_cases(T / "in_smith", [
    case("txsd;;4:18-cv-00001", "txsd", parties=[("SMITH", [("Ann Lee", info("Lee LLP", "1 Main St, Houston, TX 77002", "713-555-0101", "alee@lee.example.com"))])]),
    case("txsd;;4:18-cv-00002", "txsd", parties=[("SMITH", [("Bob Kim", info("Kim PC", "9 Elm St, Houston, TX 77002", "713-555-0202", "bkim@kim.example.com"))])]),
])
out = T / "out_smith"
run([T / "in_smith"], out)
_, m = load(out)
check("12. two 'SMITH' parties in different cases -> 2 IDs (not an acronym: no matching org initials, no division suffix)",
      len(ids_for(m, "party", "SMITH")) == 2, f"{ids_for(m, 'party', 'SMITH')}")

# --- test 13 (step 4 revision): a subsidiary suffix that is not a known acronym is not a division ---
write_cases(T / "in_grace", [
    case("paed;;2:08-cv-00001", "paed", parties=[("Q. R. GROVE & CO.", [])]),
    case("paed;;2:08-cv-00002", "paed", parties=[("Q.R. GROVE & CO.-CONN.", [])]),
])
out = T / "out_grace"
run([T / "in_grace"], out)
_, m = load(out)
g1, g2 = ids_for(m, "party", "Q. R. GROVE & CO."), ids_for(m, "party", "Q.R. GROVE & CO.-CONN.")
check("13. 'Q.R. GROVE & CO.-CONN.' and 'Q. R. GROVE & CO.' keep 2 IDs ('-CONN.' is no acronym of an org in this court)",
      len(g1) == 1 and len(g2) == 1 and g1 != g2, f"{g1} vs {g2}")

# --- test 14 (step 5): lawyers ---
nc = {"office_name": None, "address": None, "phone": None, "email": None}
def case_c(ucid, court, parties):
    c = case(ucid, court, parties=[(n, cl) for n, cl in parties])
    return c
c14 = [case_c("txsd;;4:19-cv-00001", "txsd", [("Alpha Person", [("John Q Public", nc)])]),
       case_c("txsd;;4:19-cv-00002", "txsd", [("Beta Person", [("John Q Public", nc)])]),
       case_c("txsd;;4:19-cv-00003", "txsd", [("Gamma Person", [("John Public", nc)])]),
       case_c("txsd;;4:19-cv-00004", "txsd", [("Delta Person", [("John Public", nc)])]),
       case_c("txsd;;4:19-cv-00005", "txsd", [("Acme Industries Inc", [("Mary Roe", nc)])]),
       case_c("txsd;;4:19-cv-00006", "txsd", [("Acme Industries Inc", [("Mary Roe", nc)])])]
c14.append({"ucid": "txsd;;4:19-cv-00007", "court": "txsd", "judge": None, "referred_judges": [], "docket": [],
            "parties": [{"name": "Pat Selfrep", "counsel": [{"name": "Pat Selfrep", "is_pro_se": True,
                                                              "entity_info": {"office_name": "PRO SE", "raw_info": "PRO SE"}}]}]})
write_cases(T / "in_lawyers", c14)
out = T / "out_lawyers"
run([T / "in_lawyers"], out)
_, m = load(out)
check("14a. no contact, same full name with middle initial, same court -> one ID", len(ids_for(m, "counsel", "John Q Public")) == 1,
      f"{ids_for(m, 'counsel', 'John Q Public')}")
check("14b. no contact, 'John Public' (no middle), different parties -> two IDs", len(ids_for(m, "counsel", "John Public")) == 2,
      f"{ids_for(m, 'counsel', 'John Public')}")
check("14c. no contact, same name representing the same party ID (Acme Industries) -> one ID", len(ids_for(m, "counsel", "Mary Roe")) == 1,
      f"{ids_for(m, 'counsel', 'Mary Roe')}, party {ids_for(m, 'party', 'Acme Industries Inc')}")
check("14d. pro se entry -> no counsel or firm mention", not ids_for(m, "counsel", "Pat Selfrep") and not ids_for(m, "firm", "PRO SE"),
      f"{ids_for(m, 'counsel', 'Pat Selfrep')} {ids_for(m, 'firm', 'PRO SE')}")

# --- test 15 (step 6): docket-text judges ---
def dcase(ucid, court, judge, texts):
    c = case(ucid, court, judge=judge)
    c["docket"] = [{"docket_text": t} for t in texts]
    return c
c15 = [dcase("txsd;;4:20-cv-00001", "txsd", "Zed Quill", ["Hearing held before Magistrate Judge Ann Vexx on 1/2/2020.",
                                                          "Motion hearing before Judge Quill. (jdoe)"]),
       dcase("txsd;;4:20-cv-00002", "txsd", "Zed Quill", ["ORDER signed by Judge Pat O. Lane on 2/3/2020."])]
c15 += [dcase(f"txsd;;4:20-cv-{i:05d}", "txsd", "Zed Quill", ["Case referred to Magistrate Judge Bo Krell for pretrial."]) for i in (3, 4, 5)]
write_cases(T / "in_docket", c15)
out = T / "out_docket"
run([T / "in_docket"], out)
_, m = load(out)
jn = lambda raw: {mm["entity_id"] for mm in m.values() if mm["entity_type"] == "judge" and mm["raw_name"] == raw}
check("15a. docket-only judge in 1 case, not in case fields -> no judge ID (waits in the pending pool) ('Ann Vexx')", not jn("Ann Vexx"), f"{jn('Ann Vexx')}")
check("15b. [clean-up fix 1] docket-only magistrate with a judge title in 3 different cases -> promoted to 1 ID ('Bo Krell')",
      len(jn("Bo Krell")) == 1 and sum(1 for mm in m.values() if mm["raw_name"] == "Bo Krell") == 3, f"{jn('Bo Krell')}")
check("15c. docket-only judge in 1 case, not in any case field -> no judge ID (waits in the pending pool) ('Pat O. Lane')",
      not jn("Pat O. Lane"), f"{jn('Pat O. Lane')}")
check("15d. docket surname matching the case's own judge -> kept and joins him ('Quill' -> 'Zed Quill')",
      jn("Quill") and jn("Quill") == jn("Zed Quill"), f"{jn('Quill')} vs {jn('Zed Quill')}")

# --- test 15e (clean-up): docket text attaches to a judge known from another case's fields ---
write_cases(T / "in_docket2", [dcase("txsd;;4:21-cv-00001", "txsd", "Zed Quill", []),
                               dcase("txsd;;4:21-cv-00002", "txsd", "Mae Poe", ["ORDER signed by Judge Zed Quill on 1/2/2021."])])
out = T / "out_docket2"
run([T / "in_docket2"], out)
_, m = load(out)
zq = {mm["entity_id"] for mm in m.values() if mm["entity_type"] == "judge" and mm["raw_name"] == "Zed Quill"}
zq_docket = {mm["entity_id"] for mm in m.values() if mm["entity_type"] == "judge" and mm["raw_name"] == "Zed Quill"
             and mm["ucid"] == "txsd;;4:21-cv-00002"}
check("15e. docket mention of a judge known from another case's fields -> joins that judge's ID",
      len(zq) == 1 and zq_docket == zq, f"all {zq}, docket {zq_docket}")

# --- test 16 (clean-up): type gate, roles, first-name-only judges ---
def pcase(ucid, court, parties, judge=None, docket=()):
    return {"ucid": ucid, "court": court, "judge": judge, "referred_judges": [], "docket": [{"docket_text": t} for t in docket],
            "parties": [{"name": n, "role": role, "party_type": pt,
                         "counsel": [{"name": cn, "entity_info": inf} for cn, inf in cl]} for n, role, pt, cl in parties]}
nc = {"office_name": None, "address": None, "phone": None, "email": None}
lawyer = info("Roe Law PLLC", "12 Oak St, Houston, TX 77002", "713-555-0909", "roe@roe.example.com")
c16 = [pcase("txsd;;4:22-cv-00001", "txsd", [("Acme Widgets Inc", "Plaintiff", "plaintiff", [("Ann Roe", lawyer)]),
                                            ("Bert Lowe", "Defendant", "defendant", [("Springfield Interpreter", nc), ("Jane Q Doering", nc)])]),
       pcase("txsd;;4:22-cv-00002", "txsd", [("Cara Nash", "Plaintiff", "plaintiff", [("Ann Roe", lawyer)]),
                                            ("Acme Widgets Inc", "Defendant", "defendant", [("Dee Vance", info("Attorney at Law", "3 Elm St, Houston, TX 77002", "713-555-0303", "dv@dv.example.com")),
                                                                                           ("Eli Fox", info("Eli Fox, Attorney at Law", "4 Elm St, Houston, TX 77002", "713-555-0404", "ef@ef.example.com"))]),
                                            ("Incorrectly Filed", "Defendant", "defendant", [])]),
       pcase("txsd;;4:22-cv-00003", "txsd", [("Gus Hale", "Defendant", "defendant", [])], judge="Ronnie G. Mace",
             docket=["ORDER signed by Judge Ronnie G Mace on 1/2/2022."]),
       pcase("txsd;;4:22-cv-00004", "txsd", [("Ida Moss", "Defendant", "defendant", [])], judge="Vera Klein",
             docket=["Hearing held before Judge Ronnie on 1/3/2022."])]
write_cases(T / "in_cleanup", c16)
out = T / "out_cleanup"
run([T / "in_cleanup"], out)
ents16, m = load(out)
dropped = [json.loads(l) for l in open(out / "dropped_entities.jsonl")]
dr = lambda raw: [d["reason"] for d in dropped if d["raw_name"] == raw]
check("16a. an interpreter listed as counsel is dropped (logged as court staff)",
      not ids_for(m, "counsel", "Springfield Interpreter") and any("court staff" in r for r in dr("Springfield Interpreter")),
      f"ids {ids_for(m, 'counsel', 'Springfield Interpreter')}, dropped {dr('Springfield Interpreter')}")
check("16b. a real lawyer with no contact info is kept ('Jane Q Doering')", len(ids_for(m, "counsel", "Jane Q Doering")) == 1,
      f"{ids_for(m, 'counsel', 'Jane Q Doering')}")
acme = ids_for(m, "party", "Acme Widgets Inc")
acme_ent = [e for e in ents16 if e["entity_id"] in acme]
sides = sorted(s_ for mm in m.values() if mm["raw_name"] == "Acme Widgets Inc" for s_ in mm.get("side", []))
check("16c. a party is plaintiff in case A and defendant in case B: one ID, role counts 1 + 1, each mention has role + side",
      len(acme) == 1 and acme_ent[0].get("role_counts") == {"plaintiff-side": 1, "defendant-side": 1}
      and sides == ["defendant-side", "plaintiff-side"]
      and all(mm.get("role") for mm in m.values() if mm["raw_name"] == "Acme Widgets Inc"),
      f"ids {acme}, role_counts {acme_ent[0].get('role_counts') if acme_ent else None}, sides {sides}")
ann = sorted(s_ for mm in m.values() if mm["entity_type"] == "counsel" and mm["raw_name"] == "Ann Roe" for s_ in mm.get("side", []))
check("16d. counsel mentions carry the side of the party they represent", ann == ["plaintiff-side", "plaintiff-side"], f"{ann}")
check("16e. placeholder firm 'Attorney at Law' and party 'Incorrectly Filed' dropped; firm 'Eli Fox, Attorney at Law' kept",
      not ids_for(m, "firm", "Attorney at Law") and not ids_for(m, "party", "Incorrectly Filed")
      and len(ids_for(m, "firm", "Eli Fox, Attorney at Law")) == 1 and dr("Attorney at Law") and dr("Incorrectly Filed"),
      f"{dr('Attorney at Law')} {dr('Incorrectly Filed')} {ids_for(m, 'firm', 'Eli Fox, Attorney at Law')}")
judge_single = [e["normalized_name"] for e in ents16 if e["entity_type"] == "judge" and not e.get("unresolved")
                and all(len(v.split()) == 1 for v in e["name_variants"])]
mace = ids_for(m, "judge", "Ronnie G. Mace")
mace_docket = {mm["entity_id"] for mm in m.values() if mm["entity_type"] == "judge" and mm["raw_name"] == "Ronnie"
               and mm["ucid"] == "txsd;;4:22-cv-00003"}
pending16 = [json.loads(l) for l in open(out / "pending_judges.jsonl")]
check("16f. no first-name-only judge IDs: 'Ronnie' (docket) joins Ronnie G. Mace in his case; where no Ronnie judge exists it waits in the pending pool",
      not judge_single and mace_docket == set(mace) and any(p["raw_name"] == "Ronnie" for p in pending16),
      f"single-word judges {judge_single}, Mace {mace}, docket 'Ronnie' {mace_docket}, dropped {dr('Ronnie')}")

# --- test 17 (clean-up fixes): docket magistrates, trailing words, batch independence, Stage 2 exclusion ---
def mcases(prefix, n, court="txsd"):
    return [dcase(f"{court};;4:23-cv-{prefix}{i:03d}", court, "Una Vale", [f"Hearing held before Magistrate Judge Nora Bell on 1/{i}/2023."])
            for i in range(1, n + 1)]
write_cases(T / "in_mag3", mcases("1", 3))
write_cases(T / "in_mag2", mcases("2", 2))
run([T / "in_mag3"], T / "out_mag3")
run([T / "in_mag2"], T / "out_mag2")
_, m3 = load(T / "out_mag3")
_, m2 = load(T / "out_mag2")
nb3 = {mm["entity_id"] for mm in m3.values() if mm["raw_name"] == "Nora Bell"}
nb2 = {mm["entity_id"] for mm in m2.values() if mm["raw_name"] == "Nora Bell"}
pend2 = [json.loads(l) for l in open(T / "out_mag2" / "pending_judges.jsonl")]
check("17a. a magistrate seen only in docket text in 3 cases gets 1 ID (all 3 mentions)", len(nb3) == 1
      and sum(1 for mm in m3.values() if mm["raw_name"] == "Nora Bell") == 3, f"{nb3}")
check("17b. ... in only 2 cases: no ID yet, both mentions kept in the pending pool (not dropped)",
      not nb2 and sum(1 for p in pend2 if p["raw_name"] == "Nora Bell") == 2, f"ids {nb2}, pending {len(pend2)}")
write_cases(T / "in_motion", [dcase("txsd;;4:24-cv-00001", "txsd", "Zed Quill", ["ORDER signed by Judge Zed Quill Motion granted."])])
run([T / "in_motion"], T / "out_motion")
_, mm_ = load(T / "out_motion")
zm = {x["entity_id"] for x in mm_.values() if x["raw_name"] == "Zed Quill Motion"}
zq_ = {x["entity_id"] for x in mm_.values() if x["raw_name"] == "Zed Quill"}
check("17c. '<judge> Motion' keeps the judge: the trailing word is trimmed and the mention joins Zed Quill",
      len(zq_) == 1 and zm == zq_, f"'Zed Quill Motion' -> {zm}, 'Zed Quill' -> {zq_}")
# two batches vs one batch (the 3 Nora Bell cases split 2 + 1, plus a case-field judge arriving later)
batch = mcases("3", 3) + [dcase("txsd;;4:23-cv-3999", "txsd", "Ira Penn", ["Status conference before Judge Ira Penn."])]
write_cases(T / "in_b_all", batch)
(T / "in_b1").mkdir(); (T / "in_b2").mkdir()
for i, c in enumerate(batch):
    dst = T / ("in_b1" if i < 2 else "in_b2")
    (dst / (c["ucid"].replace(";;", "-").replace(":", "-") + ".json")).write_text(json.dumps(c))
run([T / "in_b_all"], T / "out_b_one")
run([T / "in_b1"], T / "out_b_two")
run([T / "in_b2"], T / "out_b_two")
def judge_groups(d):
    _, mx = load(d)
    g = collections.defaultdict(set)
    for x in mx.values():
        if x["entity_type"] == "judge":
            g[x["entity_id"]].add((x["ucid"], x["raw_name"]))
    return sorted(sorted(v) for v in g.values())
check("17d. the same files in two batches vs one batch give the same judges", judge_groups(T / "out_b_one") == judge_groups(T / "out_b_two"),
      f"one {judge_groups(T / 'out_b_one')} | two {judge_groups(T / 'out_b_two')}")
import importlib.util
spec = importlib.util.spec_from_file_location("v3m", SCRIPT); V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
rec = {"entity_type": "counsel", "raw_name": "Federal Public Defender", "clean_name": "federal public defender", "_ctx": collections.defaultdict(list)}
for i in range(30):
    V.INDEX.counsel_name_cases["federal public defender"].add(f"c{i}")
rec2 = {"entity_type": "counsel", "raw_name": "Kim Q Lake", "clean_name": "kim q lake", "_ctx": collections.defaultdict(list, {"emails": ["k@x.example.com"]})}
check("17e. office names and counsel with contact info never go to Stage 2",
      V.type_gate_stage1(rec)[0] == "keep" and V.type_gate_stage1(rec2)[0] == "keep", f"{V.type_gate_stage1(rec)} {V.type_gate_stage1(rec2)}")

# --- test 18 (option A round): first-name join only when exactly one judge in the case has that first name ---
c18 = case("txsd;;4:25-cv-00001", "txsd", judge="Ann Lee", referred=["Ann Moss"])
c18["docket"] = [{"docket_text": "Hearing held before Judge Ann on 1/2/2025."}]
c18b = case("txsd;;4:25-cv-00002", "txsd", judge="Bea Holt")
c18b["docket"] = [{"docket_text": "Hearing held before Judge Bea on 1/3/2025."}]
write_cases(T / "in_firstname2", [c18, c18b])
run([T / "in_firstname2"], T / "out_firstname2")
_, m = load(T / "out_firstname2")
pend18 = [json.loads(l) for l in open(T / "out_firstname2" / "pending_judges.jsonl")]
bea = {x["entity_id"] for x in m.values() if x["raw_name"] in ("Bea", "Bea Holt")}
check("18. first-name join: 'Bea' joins the only Bea judge in its case; 'Ann' with two Ann judges in the case stays pending",
      len(bea) == 1 and not any(x["raw_name"] == "Ann" for x in m.values()) and any(p["raw_name"] == "Ann" for p in pend18),
      f"Bea {bea}, Ann pending {[p['raw_name'] for p in pend18]}")

# --- test 5: incremental 800 + 200 on the pilot (needs V3_PILOT_DIR) ---
if PILOT is None:
    print("SKIP  5a-5d. incremental 800 + 200 (set V3_PILOT_DIR to a folder of 1,000 PACER JSON files)")
    print(f"\n{sum(ok for _, ok in results)}/{len(results)} passed (test 5 skipped)")
    sys.exit(0 if all(ok for _, ok in results) else 1)
names = sorted(p.name for p in PILOT.glob("*.json"))
for sub, part in (("in_800", names[:800]), ("in_200", names[800:])):
    (T / sub).mkdir()
    for n in part:
        (T / sub / n).symlink_to(PILOT / n)
out = T / "out_800_200"
run([T / "in_800"], out)
ents800, m800 = load(out)
snap_m = {k: v["entity_id"] for k, v in m800.items()}
snap_e = {e["entity_id"]: (e["entity_type"], e.get("sjid") or e.get("spid") or e.get("scid") or e.get("sfid")) for e in ents800}
run([T / "in_200"], out)
ents, m = load(out)
kept_m = sum(m[k]["entity_id"] == v for k, v in snap_m.items())
now_e = {e["entity_id"]: (e["entity_type"], e.get("sjid") or e.get("spid") or e.get("scid") or e.get("sfid")) for e in ents}
kept_e = sum(now_e.get(k) == v for k, v in snap_e.items())
state = json.load(open(out / "state.json"))
check("5a. 800 + 200: old mentions keep their entity ID", kept_m == len(snap_m), f"{kept_m}/{len(snap_m)}")
check("5b. 800 + 200: old entity IDs all still present, same type and ID", kept_e == len(snap_e), f"{kept_e}/{len(snap_e)}")
check("5c. 800 + 200: 1,000 files recorded", len(state["processed_files"]) == 1000, f"{len(state['processed_files'])}")
before = {f: hashlib.sha256((out / f).read_bytes()).hexdigest() for f in ("entities.jsonl", "mentions.jsonl", "state.json", "uncertain_pairs.jsonl", "dropped_entities.jsonl", "name_gate_cache.jsonl", "gate_review.jsonl", "pending_judges.jsonl")}
run([PILOT / names[0]], out)
after = {f: hashlib.sha256((out / f).read_bytes()).hexdigest() for f in before}
check("5d. re-adding an already processed file changes nothing", before == after, "all 8 output files byte-identical" if before == after else "files changed")

print(f"\n{sum(ok for _, ok in results)}/{len(results)} passed")
sys.exit(0 if all(ok for _, ok in results) else 1)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU-only deterministic builder for AX06-AX15 prospective extension inputs.
It DOES NOT load any LLM and DOES NOT generate model outputs.

Scientific content source:
  202A_AX06_AX15_INPUT_SOURCE_MASTER_FROZEN_V1.tsv

Protocol inheritance:
  runs/multibranch_kg_v2_grounded/20_AX_Prospective_Replication_v1/00_FROZEN_INPUTS/
  MultiBranch_LLM_Experiment_v1/03_SCRIPTS/run_hf.py

Design:
- AX06-AX15 remain strict A1-A5 items.
- Flat/Multi receive identical content atoms.
- Multi preserves branch grouping.
- Flat removes grouping and uses deterministic continuation of the original
  random.Random(20260808) AX01->AX05 shuffle stream.
- No legacy ensemble probability/vote metadata is invented for Phase2-recovered evidence.
- New CLAIM atoms are explicitly marked in the manifest as frozen curated semantic
  paraphrases from the final strict benchmark, not verbatim quotations.
"""

from pathlib import Path
import argparse, csv, hashlib, json, random, re, shutil, sys

SEED = 20260808
EXPECTED_RUNNER_SHA256 = "40d62d5f6fc72d6107e78501a597acd4ca997bd67c3ea22c3e6941d895b9e05a"
OLD_DIR_REL = Path("runs/multibranch_kg_v2_grounded/20_AX_Prospective_Replication_v1/00_FROZEN_INPUTS")
OUT_REL = Path("runs/multibranch_kg_v2_grounded/35_FinalEnhanced_AX06_AX15_Prospective_Extension_v1/00_FROZEN_INPUTS")
RUNNER_REL = Path("MultiBranch_LLM_Experiment_v1/03_SCRIPTS/run_hf.py")
SOURCE_MASTER_NAME = "202A_AX06_AX15_INPUT_SOURCE_MASTER_FROZEN_V1.tsv"

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def read_jsonl(path):
    rows=[]
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def atom_ids(context):
    return re.findall(r"\[(S\d+)\]", context)

def clean(v):
    return "" if v is None else str(v).strip()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--source-master", default=SOURCE_MASTER_NAME)
    args=ap.parse_args()

    root=Path(args.root).resolve()
    old_dir=root/OLD_DIR_REL
    runner=root/RUNNER_REL
    source_master=Path(args.source_master)
    if not source_master.is_absolute():
        source_master=root/source_master

    required=[old_dir/"flat_kg.jsonl", old_dir/"multibranch_kg.jsonl",
              old_dir/"90_FROZEN_MANIFEST.json", runner, source_master]
    missing=[str(p) for p in required if not p.exists()]
    if missing:
        print("ERROR missing required files:")
        for p in missing: print(" -", p)
        sys.exit(2)

    runner_hash=sha256_file(runner)
    if runner_hash != EXPECTED_RUNNER_SHA256:
        print("ERROR runner hash drift")
        print("EXPECTED", EXPECTED_RUNNER_SHA256)
        print("ACTUAL  ", runner_hash)
        sys.exit(3)

    old_flat=read_jsonl(old_dir/"flat_kg.jsonl")
    old_multi=read_jsonl(old_dir/"multibranch_kg.jsonl")
    if len(old_flat)!=5 or len(old_multi)!=5:
        raise SystemExit("ERROR old AX01-AX05 inputs are not 5+5.")

    # Exact inherited prompts from old AX prospective replication.
    system_prompt=old_flat[0]["system_prompt"]
    old_q=old_flat[0]["question"]
    old_ctx=old_flat[0]["context"]
    user_template=old_flat[0]["user_prompt"]
    if old_q not in user_template or old_ctx not in user_template:
        raise SystemExit("ERROR could not derive exact user prompt template.")
    user_template=user_template.replace(old_q, "{QUESTION}", 1).replace(old_ctx, "{CONTEXT}", 1)

    # Reproduce/consume AX01-AX05 shuffle stream exactly.
    rng=random.Random(SEED)
    old_shuffle_audit=[]
    for f,m in zip(old_flat, old_multi):
        mids=atom_ids(m["context"])
        actual=atom_ids(f["context"])
        expected=mids.copy()
        rng.shuffle(expected)
        ok=(expected==actual)
        old_shuffle_audit.append((f["benchmark_id"], mids, expected, actual, ok))
    if not all(x[-1] for x in old_shuffle_audit):
        raise SystemExit("ERROR original AX01-AX05 shuffle stream not reproduced.")

    # Read frozen source master.
    with open(source_master, encoding="utf-8-sig", newline="") as f:
        rows=list(csv.DictReader(f, delimiter="\t"))
    expected_ax=[f"AX{i:02d}" for i in range(6,16)]
    got=[r["AX_ID"] for r in rows]
    if got != expected_ax:
        raise SystemExit(f"ERROR AX order mismatch: {got}")

    # Strict-status guardrails.
    for r in rows:
        for col in ["A1","A2","A3","A4_A","A4_B","A5_A","A5_B"]:
            val=clean(r.get(col)).upper()
            if "PASS" not in val and "VERIFIED" not in val:
                raise SystemExit(f"ERROR {r['AX_ID']} {col} not PASS/VERIFIED: {r.get(col)}")
        if "PASS" not in clean(r.get("Strict_Status")).upper():
            raise SystemExit(f"ERROR {r['AX_ID']} Strict_Status not PASS: {r.get('Strict_Status')}")

    gold=[]
    profiles=[]
    flat_rows=[]
    multi_rows=[]
    stats=[]
    audit_rows=[]

    for r in rows:
        ax=r["AX_ID"]
        bid=r["Benchmark_ID"]
        q=clean(r["Authoritative_Question"])
        dep=clean(r["Deposit_Name_For_Input"])
        dim=clean(r["Dimension_For_Input"])
        model_a=clean(r["Model_A"])
        model_b=clean(r["Model_B"])
        eva=clean(r["Verified_Evidence_A"])
        evb=clean(r["Verified_Evidence_B"])
        srca=clean(r["Source_A"])
        srcb=clean(r["Source_B"])
        cand=clean(r["Candidate_ID"])

        if not all([bid,q,dep,dim,model_a,model_b,eva,evb,cand]):
            raise SystemExit(f"ERROR missing core field in {ax}")

        # Six atoms, 3 per branch. CLAIM is a frozen curated semantic paraphrase,
        # not represented as a verbatim quotation.
        atoms=[
            {"id":"S01","branch":1,"kind":"SUMMARY","text":model_a},
            {"id":"S02","branch":1,"kind":"CLAIM","text":model_a},
            {"id":"S03","branch":1,"kind":"EVIDENCE","text":eva,"source":srca},
            {"id":"S04","branch":2,"kind":"SUMMARY","text":model_b},
            {"id":"S05","branch":2,"kind":"CLAIM","text":model_b},
            {"id":"S06","branch":2,"kind":"EVIDENCE","text":evb,"source":srcb},
        ]

        def atom_line(a):
            s=f"[{a['id']}] {a['kind']}: {a['text']}"
            if a["kind"]=="EVIDENCE" and a.get("source"):
                s += f" | source={a['source']}"
            return s

        multi_context = (
            "[BRANCH 1]\n" + "\n".join(atom_line(a) for a in atoms[:3]) +
            "\n[BRANCH 2]\n" + "\n".join(atom_line(a) for a in atoms[3:])
        )

        shuffled=atoms.copy()
        rng.shuffle(shuffled)
        flat_context="\n".join(atom_line(a) for a in shuffled)

        # Exact same user prompt template as AX01-AX05.
        flat_user=user_template.replace("{QUESTION}", q).replace("{CONTEXT}", flat_context)
        multi_user=user_template.replace("{QUESTION}", q).replace("{CONTEXT}", multi_context)

        common={"benchmark_id":bid,"pool":"AX","question":q,"deposit_name":dep,
                "dimension":dim,"system_prompt":system_prompt}
        fr=dict(common)
        fr.update({"condition":"FLAT_KG","context":flat_context,"user_prompt":flat_user})
        mr=dict(common)
        mr.update({"condition":"MULTIBRANCH_KG","context":multi_context,"user_prompt":multi_user})
        flat_rows.append(fr); multi_rows.append(mr)

        # Gold: no invented ensemble statistics for recovered manual evidence.
        branches=[]
        for bidx,(label,summary,evidence,source) in enumerate([
            ("MODEL_A",model_a,eva,srca),("MODEL_B",model_b,evb,srcb)
        ], start=1):
            branches.append({
                "label": label,
                "branch_id": f"BRANCH::{bid}::B{bidx}",
                "semantic_summary": summary,
                "source_claim": summary,
                "source_claim_status": "FROZEN_CURATED_SEMANTIC_PARAPHRASE_NOT_VERBATIM_QUOTE",
                "support_count": 1,
                "high_conf_support_count": None,
                "mean_support_probability": None,
                "evidence": [{
                    "evidence_id": f"EVIDENCE::{bid}::B{bidx}::VERIFIED1",
                    "text": evidence,
                    "source_reference": source,
                    "support_validity": "VERIFIED_SUPPORT",
                    "supports_probability_mean": None,
                    "supports_seed_votes": None,
                    "ensemble_vote_tier": "NOT_APPLICABLE_PHASE2_MANUAL_RECOVERY"
                }]
            })
        gold.append({
            "benchmark_id":bid,"pool":"AX","deposit_name":dep,"dimension":dim,
            "question":q,"gold_branch_relation":"COMPETING",
            "curation_status":"FROZEN_GOLD_AX_EXTENSION_V1",
            "source_question_group_id":cand,
            "notes":"Protocol-preserving strict A1-A5 extension. AX06-AX15 finalized before generating their model outputs. Do not claim AX01-AX15 were all preregistered as one initial set.",
            "branches":branches,"paper_count":None,"paper_ids":[],
            "semantic_branch_count":2,
            "all_branches_have_high_conf_evidence":None,
            "freeze_date":"2026-08-29",
            "freeze_note":"Phase2 strict PASS; evidence text comes from final manually verified master. No legacy ensemble probability/vote metadata invented."
        })
        profiles.append({
            "benchmark_id":bid,"source_question_group_id":cand,
            "screening_criteria":"A1_A5",
            "relation_profile":{"competing":True,"complementary":False,"convergent":False,"synthesis":False},
            "primary_relation":"COMPETING",
            "core_semantic_branches":[
                {"label":"MODEL_A","branch_id":f"BRANCH::{bid}::B1",
                 "support_validity":"VERIFIED_SUPPORT",
                 "discriminating_role":"NOT_PRECOMMITTED_IN_EXTENSION_INPUT_BUILD",
                 "human_verification_provenance":clean(r.get("Author_Confirmation")) or "AUTHOR_MANUAL_A5_CONFIRMED",
                 "source_reference":srca},
                {"label":"MODEL_B","branch_id":f"BRANCH::{bid}::B2",
                 "support_validity":"VERIFIED_SUPPORT",
                 "discriminating_role":"NOT_PRECOMMITTED_IN_EXTENSION_INPUT_BUILD",
                 "human_verification_provenance":clean(r.get("Author_Confirmation")) or "AUTHOR_MANUAL_A5_CONFIRMED",
                 "source_reference":srcb},
            ],
            "reporting_scope":"PROTOCOL_PRESERVING_PROSPECTIVE_EXTENSION",
            "prohibited_claim":"DO_NOT_DESCRIBE_AX01_AX15_AS_SINGLE_INITIAL_PREREGISTERED_SET"
        })

        flat_ids=atom_ids(flat_context)
        multi_ids=atom_ids(multi_context)
        same=set(flat_ids)==set(multi_ids) and len(flat_ids)==len(multi_ids)==6
        stats.append({
            "benchmark_id":bid,"content_atoms":6,
            "flat_context_chars":len(flat_context),
            "multi_context_chars":len(multi_context),
            "same_content_atoms":same,
            "flat_atom_order":" ".join(flat_ids),
            "multi_atom_order":" ".join(multi_ids)
        })
        audit_rows.append((ax,bid,cand,same," ".join(flat_ids)))

    out=root/OUT_REL
    if out.exists():
        raise SystemExit(f"ERROR output directory already exists: {out}\nRefusing to overwrite a frozen build.")
    out.mkdir(parents=True)

    # Copy frozen scientific source master into build directory.
    shutil.copy2(source_master, out/SOURCE_MASTER_NAME)

    write_jsonl(out/"benchmark_gold_AX06_AX15_v1.jsonl", gold)
    write_jsonl(out/"benchmark_AX06_AX15_relation_evidence_profile_v1.jsonl", profiles)
    write_jsonl(out/"flat_kg.jsonl", flat_rows)
    write_jsonl(out/"multibranch_kg.jsonl", multi_rows)

    with open(out/"context_stats_AX06_AX15_v1.tsv","w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(stats[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(stats)

    manifest={
        "status":"FROZEN_AX06_AX15_EXTENSION_INPUTS_V1",
        "freeze_date":"2026-08-29",
        "benchmark_items":10,
        "benchmark_ids":[r["Benchmark_ID"] for r in rows],
        "prospective_numbering":{r["AX_ID"]:r["Candidate_ID"] for r in rows},
        "screening_criteria":"A1_A5_UNCHANGED",
        "historical_benchmark_modified":False,
        "historical_experiment_rerun":False,
        "conditions_to_generate":["FLAT_KG","MULTIBRANCH_KG"],
        "expected_outputs":20,
        "generation_protocol":{
            "model_family":"Qwen2.5-7B",
            "model_argument":"/root/autodl-tmp/models/Qwen2.5-7B-Instruct-ms",
            "temperature":0.0,"do_sample":False,"seed":SEED,"max_new_tokens":900,
            "runner_reference":str(RUNNER_REL),
            "runner_sha256":runner_hash,
            "system_prompt_inherited_exactly_from_AX01_AX05":True,
            "user_prompt_template_inherited_exactly_from_AX01_AX05":True
        },
        "flat_multi_manipulation":{
            "same_content_atoms":True,
            "difference":"Flat removes branch grouping and shuffles atoms; Multi preserves branch grouping.",
            "shuffle_algorithm":"Python random.Random(20260808), deterministic continuation after consuming original AX01->AX05 shuffle stream.",
            "original_AX01_AX05_shuffle_reproduced":True
        },
        "gold_policy":{
            "branch_relation":"COMPETING",
            "evidence_rule":"Manually VERIFIED_SUPPORT from frozen final AX master.",
            "legacy_ensemble_metadata":"NOT_INVENTED_WHEN_UNAVAILABLE",
            "claim_atom_policy":"Frozen curated semantic paraphrase; not represented as verbatim quotation."
        },
        "temporal_transparency":{
            "initial_set":"AX01-AX05",
            "extension":"AX06-AX15",
            "extension_occurred_after_initial_AX01_AX05_results_were_observed":True,
            "extension_criteria_changed":False,
            "AX06_AX15_outputs_seen_before_this_freeze":False,
            "prohibited_wording":"Do not state that all AX01-AX15 were a single initial preregistered benchmark."
        },
        "reporting_guardrails":[
            "Main Experiment I remains frozen and is not rerun.",
            "Historical A01-A03 remain separate from prospective strict AX.",
            "AX01-AX15 may be presented once as the final prospective strict set n=15, while Methods discloses the protocol-preserving extension timing.",
            "Primary paired contrast is Flat-KG vs Multi-branch KG."
        ],
        "source_master_sha256":sha256_file(out/SOURCE_MASTER_NAME),
        "old_AX01_AX05_input_manifest_sha256":sha256_file(old_dir/"90_FROZEN_MANIFEST.json")
    }
    with open(out/"90_FROZEN_MANIFEST.json","w",encoding="utf-8") as f:
        json.dump(manifest,f,ensure_ascii=False,indent=2)

    with open(out/"91_BUILD_AUDIT.txt","w",encoding="utf-8") as f:
        f.write("AX06-AX15 FROZEN INPUT BUILD AUDIT\n")
        f.write("="*72+"\n")
        f.write("CPU_ONLY=TRUE\n")
        f.write("NO_MODEL_GENERATION=TRUE\n")
        f.write("OLD_AX01_AX05_SHUFFLE_REPRODUCED=TRUE\n")
        f.write("RUNNER_SHA256_MATCH=TRUE\n")
        f.write("ITEMS=10\nFLAT_INPUTS=10\nMULTI_INPUTS=10\n")
        f.write("ALL_SAME_CONTENT_ATOMS="+str(all(s["same_content_atoms"] for s in stats)).upper()+"\n")
        f.write("CLAIM_ATOMS=FROZEN_CURATED_SEMANTIC_PARAPHRASES_NOT_VERBATIM_QUOTES\n")
        f.write("LEGACY_ENSEMBLE_METADATA_INVENTED=FALSE\n\n")
        for ax,bid,cand,same,order in audit_rows:
            f.write(f"{ax}\t{cand}\t{bid}\tPARITY={same}\tFLAT_ORDER={order}\n")

    # Hash all frozen outputs except SHA file itself.
    files=sorted(p for p in out.iterdir() if p.is_file() and p.name!="SHA256SUMS.txt")
    with open(out/"SHA256SUMS.txt","w",encoding="utf-8") as f:
        for p in files:
            f.write(f"{sha256_file(p)}  {p.name}\n")

    print("SUCCESS")
    print("OUTPUT_DIR=", out)
    print("ITEMS=10")
    print("FLAT_INPUTS=10")
    print("MULTI_INPUTS=10")
    print("EXPECTED_MODEL_OUTPUTS=20")
    print("ALL_SAME_CONTENT_ATOMS=", all(s["same_content_atoms"] for s in stats))
    print("OLD_AX01_AX05_SHUFFLE_REPRODUCED=True")
    print("RUNNER_SHA256_MATCH=True")
    print("GPU_REQUIRED=False")

if __name__=="__main__":
    main()

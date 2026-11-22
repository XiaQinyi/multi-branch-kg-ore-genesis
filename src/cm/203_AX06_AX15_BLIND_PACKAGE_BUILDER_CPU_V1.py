#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build blinded evaluation package for AX06-AX15 (20 frozen responses).
CPU only. Does not modify or regenerate model outputs.

Outputs:
35_FinalEnhanced_AX06_AX15_Prospective_Extension_v1/
  02_BLIND_EVAL_PRECOMMITTED/
  02_BLIND_EVAL_SECRET/
  03_BLIND_REVIEW_READY/
"""
from pathlib import Path
import argparse, csv, hashlib, json, random, re, shutil, sys

SEED = 20260829
BLIND_START = 11

BASE_REL = Path("runs/multibranch_kg_v2_grounded/35_FinalEnhanced_AX06_AX15_Prospective_Extension_v1")
OLD_REL = Path("runs/multibranch_kg_v2_grounded/20_AX_Prospective_Replication_v1")
IN_REL = BASE_REL / "00_FROZEN_INPUTS"
OUT_REL = BASE_REL / "01_FROZEN_OUTPUTS"
PRE_REL = BASE_REL / "02_BLIND_EVAL_PRECOMMITTED"
SEC_REL = BASE_REL / "02_BLIND_EVAL_SECRET"
READY_REL = BASE_REL / "03_BLIND_REVIEW_READY"

def read_jsonl(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

def write_jsonl(p, rows):
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def parse_registry(multi_context):
    rows=[]
    branch=None
    for line in multi_context.splitlines():
        line=line.strip()
        if not line:
            continue
        if line.startswith("[BRANCH "):
            branch=line.strip("[]")
            continue
        m=re.match(r'^\[(S\d+)\]\s+([A-Z]+):\s*(.*)$', line)
        if not m:
            continue
        sid, kind, text=m.groups()
        rows.append({"source_id":sid,"kind":kind,"text":text})
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".")
    args=ap.parse_args()
    root=Path(args.root).resolve()

    inp=root/IN_REL
    out=root/OUT_REL
    pre=root/PRE_REL
    sec=root/SEC_REL
    ready=root/READY_REL
    old=root/OLD_REL

    required=[
        inp/"benchmark_gold_AX06_AX15_v1.jsonl",
        inp/"multibranch_kg.jsonl",
        out/"10_FLAT_KG_OUTPUTS_10.jsonl",
        out/"11_MULTIBRANCH_KG_OUTPUTS_10.jsonl",
        out/"90_FROZEN_MANIFEST.json",
        old/"02_BLIND_EVAL_PRECOMMITTED/20_SCORE_SHEET_RATER1_10x6.tsv",
        old/"02_BLIND_EVAL_PRECOMMITTED/30_SEMANTIC_EVALUATION_RUBRIC_FROZEN_COPY.md",
        old/"02_BLIND_EVAL_PRECOMMITTED/31_SCORING_RANGE_AND_BLINDING_NOTE.txt",
    ]
    missing=[str(p) for p in required if not p.exists()]
    if missing:
        print("ERROR missing:")
        for p in missing: print(" -",p)
        sys.exit(2)

    # Refuse overwrite: once blinded, freeze it.
    for d in [pre,sec,ready]:
        if d.exists():
            raise SystemExit(f"ERROR refusing to overwrite existing blind directory: {d}")

    gold={r["benchmark_id"]:r for r in read_jsonl(inp/"benchmark_gold_AX06_AX15_v1.jsonl")}
    multi_in={r["benchmark_id"]:r for r in read_jsonl(inp/"multibranch_kg.jsonl")}
    flat=read_jsonl(out/"10_FLAT_KG_OUTPUTS_10.jsonl")
    multi=read_jsonl(out/"11_MULTIBRANCH_KG_OUTPUTS_10.jsonl")

    if len(flat)!=10 or len(multi)!=10:
        raise SystemExit("ERROR expected 10 flat + 10 multi responses")

    cells=flat+multi
    # Validate 20 unique benchmark-condition cells
    keys=[(r.get("benchmark_id"),r.get("condition")) for r in cells]
    if len(set(keys))!=20:
        raise SystemExit("ERROR duplicate benchmark-condition cells")

    # Deterministic blind shuffle, performed only after outputs frozen.
    rng=random.Random(SEED)
    rng.shuffle(cells)

    pre.mkdir(parents=True)
    sec.mkdir(parents=True)
    ready.mkdir(parents=True)

    blind_shell=[]
    secret=[]
    ready_rows=[]

    for idx,r in enumerate(cells, start=BLIND_START):
        blind_id=f"AXR{idx:03d}"
        bid=r["benchmark_id"]
        if bid not in gold or bid not in multi_in:
            raise SystemExit(f"ERROR missing gold/input for {bid}")

        g=gold[bid]
        branches=g.get("branches",[])
        if len(branches)!=2:
            raise SystemExit(f"ERROR {bid} gold branches !=2")

        registry=parse_registry(multi_in[bid]["context"])
        if len(registry)!=6:
            raise SystemExit(f"ERROR {bid} neutral source registry !=6 atoms")

        shell={
            "blind_id":blind_id,
            "question":g["question"],
            "gold_relation":g["gold_branch_relation"],
            "gold_branch_1":branches[0]["semantic_summary"],
            "gold_branch_2":branches[1]["semantic_summary"],
            "condition_neutral_source_registry":registry
        }
        blind_shell.append(shell)

        secret.append({
            "blind_id":blind_id,
            "benchmark_id":bid,
            "condition":r["condition"]
        })

        # This is what the reviewer actually sees. No condition/benchmark id.
        ready_row=dict(shell)
        ready_row["model_response"]=r["raw_output"]
        ready_rows.append(ready_row)

    # Precommitted blind shell
    write_jsonl(pre/"10_BLIND_REVIEW_SHELL_20.jsonl", blind_shell)

    # Use exact old score-sheet header order.
    old_score=old/"02_BLIND_EVAL_PRECOMMITTED/20_SCORE_SHEET_RATER1_10x6.tsv"
    with old_score.open("r",encoding="utf-8-sig") as f:
        header=f.readline().rstrip("\r\n").split("\t")
    if not header or header[0]!="blind_id":
        raise SystemExit("ERROR could not inherit old score sheet schema")

    def score_sheet(path):
        with path.open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=header,delimiter="\t",extrasaction="ignore")
            w.writeheader()
            for row in blind_shell:
                d={k:"" for k in header}
                d["blind_id"]=row["blind_id"]
                if "review_status" in d:
                    d["review_status"]="PENDING"
                w.writerow(d)

    score_sheet(pre/"20_SCORE_SHEET_RATER1_20x6.tsv")
    score_sheet(pre/"21_SCORE_SHEET_RATER2_20x6.tsv")

    # Copy exact frozen rubric/note from AX01-AX05.
    shutil.copy2(old/"02_BLIND_EVAL_PRECOMMITTED/30_SEMANTIC_EVALUATION_RUBRIC_FROZEN_COPY.md",
                 pre/"30_SEMANTIC_EVALUATION_RUBRIC_FROZEN_COPY.md")
    shutil.copy2(old/"02_BLIND_EVAL_PRECOMMITTED/31_SCORING_RANGE_AND_BLINDING_NOTE.txt",
                 pre/"31_SCORING_RANGE_AND_BLINDING_NOTE.txt")

    # Secret mapping
    write_jsonl(sec/"blind_condition_key_SECRET.jsonl", secret)

    # Review-ready package
    write_jsonl(ready/"10_BLIND_REVIEW_READY_20.jsonl", ready_rows)

    with (ready/"11_BLIND_REVIEW_READY_20.tsv").open("w",encoding="utf-8",newline="") as f:
        fields=["blind_id","question","gold_relation","gold_branch_1","gold_branch_2",
                "condition_neutral_source_registry_json","model_response"]
        w=csv.DictWriter(f,fieldnames=fields,delimiter="\t")
        w.writeheader()
        for r in ready_rows:
            w.writerow({
                "blind_id":r["blind_id"],
                "question":r["question"],
                "gold_relation":r["gold_relation"],
                "gold_branch_1":r["gold_branch_1"],
                "gold_branch_2":r["gold_branch_2"],
                "condition_neutral_source_registry_json":json.dumps(r["condition_neutral_source_registry"],ensure_ascii=False),
                "model_response":r["model_response"]
            })

    score_sheet(ready/"20_SCORE_SHEET_RATER1_20x6.tsv")
    score_sheet(ready/"21_SCORE_SHEET_RATER2_20x6.tsv")
    shutil.copy2(pre/"30_SEMANTIC_EVALUATION_RUBRIC_FROZEN_COPY.md",
                 ready/"30_SEMANTIC_EVALUATION_RUBRIC_FROZEN_COPY.md")
    shutil.copy2(pre/"31_SCORING_RANGE_AND_BLINDING_NOTE.txt",
                 ready/"31_SCORING_RANGE_AND_BLINDING_NOTE.txt")

    # Manifests
    pre_manifest={
        "status":"FROZEN_AX06_AX15_BLIND_EVAL_PRECOMMITTED_V1",
        "freeze_date":"2026-08-29",
        "responses":20,
        "blind_ids":"AXR011-AXR030",
        "blinding_seed":SEED,
        "blinding_performed_after_raw_output_freeze":True,
        "condition_hidden_from_review_package":True,
        "benchmark_id_hidden_from_review_package":True,
        "rubric_inherited_exactly_from_AX01_AX05":True,
        "score_schema_inherited_exactly_from_AX01_AX05":True,
        "source_registry":"condition-neutral Sxx registry reconstructed from MULTIBRANCH input; same content atoms were supplied to Flat and Multi.",
        "no_model_regeneration":True
    }
    (pre/"90_FROZEN_MANIFEST.json").write_text(json.dumps(pre_manifest,ensure_ascii=False,indent=2),encoding="utf-8")

    sec_manifest={
        "status":"SECRET_CONDITION_KEY_AX06_AX15_V1",
        "responses":20,
        "blind_ids":"AXR011-AXR030",
        "DO_NOT_SHARE_WITH_RATER":True
    }
    (sec/"90_SECRET_MANIFEST.json").write_text(json.dumps(sec_manifest,ensure_ascii=False,indent=2),encoding="utf-8")

    ready_manifest={
        "status":"BLIND_REVIEW_READY_AX06_AX15_V1",
        "responses":20,
        "condition_labels_present":False,
        "benchmark_ids_present":False,
        "model_response_present":True,
        "gold_reference_present":True,
        "ready_for_blinded_semantic_scoring":True
    }
    (ready/"90_FROZEN_MANIFEST.json").write_text(json.dumps(ready_manifest,ensure_ascii=False,indent=2),encoding="utf-8")

    # hashes
    for d in [pre,sec,ready]:
        files=sorted(p for p in d.iterdir() if p.is_file() and p.name!="SHA256SUMS.txt")
        with (d/"SHA256SUMS.txt").open("w",encoding="utf-8") as f:
            for p in files:
                f.write(f"{sha(p)}  {p.name}\n")

    # Final safety audit: ready file must not contain condition names or benchmark IDs.
    ready_text=(ready/"10_BLIND_REVIEW_READY_20.jsonl").read_text(encoding="utf-8")
    leaks=[]
    for token in ["FLAT_KG","MULTIBRANCH_KG"]:
        if token in ready_text: leaks.append(token)
    for bid in gold:
        if bid in ready_text: leaks.append(bid)

    print("SUCCESS")
    print("RESPONSES=20")
    print("BLIND_IDS=AXR011-AXR030")
    print("SECRET_KEY_ROWS=20")
    print("RATER1_SCORE_ROWS=20")
    print("RATER2_SCORE_ROWS=20")
    print("CONDITION_LEAKS=",len(leaks))
    print("READY_DIR=",ready)
    print("GPU_REQUIRED=False")
    if leaks:
        print("LEAK_TOKENS=",leaks)
        sys.exit(4)

if __name__=="__main__":
    main()

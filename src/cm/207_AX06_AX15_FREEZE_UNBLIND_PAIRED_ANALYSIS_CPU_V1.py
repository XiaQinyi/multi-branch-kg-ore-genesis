#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Freeze uploaded Rater-1 blind scores, then unblind AX06-AX15 and compute paired Flat-vs-Multi results.
CPU only. No model loading. No regeneration.

Expected uploaded files in project root:
- 204_RATER1_BLINDED_SCORES_AXR011_AXR030_FROZEN_V1.tsv
- 206_RATER1_BLINDED_SCORING_QC_V1.json
"""

from pathlib import Path
import argparse, csv, json, hashlib, shutil, sys

BASE_REL = Path("runs/multibranch_kg_v2_grounded/35_FinalEnhanced_AX06_AX15_Prospective_Extension_v1")
SECRET_REL = BASE_REL / "02_BLIND_EVAL_SECRET/blind_condition_key_SECRET.jsonl"
FREEZE_REL = BASE_REL / "04_RATER1_SCORE_FREEZE"
UNBLIND_REL = BASE_REL / "05_UNBLINDED_RATER1_RESULTS_FROZEN"

SCORE_FILE = "204_RATER1_BLINDED_SCORES_AXR011_AXR030_FROZEN_V1.tsv"
QC_FILE = "206_RATER1_BLINDED_SCORING_QC_V1.json"

METRICS = [
    "branch_coverage",
    "collapse",
    "relation_awareness",
    "unsupported",
    "uncertainty",
    "evidence_attribution",
]

def sha256(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def read_jsonl(p):
    rows=[]
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def read_tsv(p):
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))

def write_tsv(p, rows, fields):
    with p.open("w", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def as_float(v):
    if v is None or str(v).strip()=="":
        return None
    return float(v)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".")
    args=ap.parse_args()
    root=Path(args.root).resolve()

    score_src=root/SCORE_FILE
    qc_src=root/QC_FILE
    secret=root/SECRET_REL
    freeze_dir=root/FREEZE_REL
    unblind_dir=root/UNBLIND_REL

    for p in [score_src, qc_src, secret]:
        if not p.exists():
            raise SystemExit(f"ERROR missing required file: {p}")

    if freeze_dir.exists() or unblind_dir.exists():
        raise SystemExit("ERROR target freeze/unblind directory already exists; refusing overwrite.")

    scores=read_tsv(score_src)
    if len(scores)!=20:
        raise SystemExit(f"ERROR score rows={len(scores)}, expected 20")

    expected_ids=[f"AXR{i:03d}" for i in range(11,31)]
    got_ids=[r["blind_id"] for r in scores]
    if sorted(got_ids)!=expected_ids:
        raise SystemExit(f"ERROR blind IDs mismatch: {sorted(got_ids)}")

    for r in scores:
        if r.get("review_status","").strip().upper()!="DONE":
            raise SystemExit(f"ERROR {r['blind_id']} review_status not DONE")
        for m in METRICS:
            if as_float(r.get(m)) is None:
                raise SystemExit(f"ERROR {r['blind_id']} missing metric {m}")

    # Freeze blind scores BEFORE opening/using secret key.
    freeze_dir.mkdir(parents=True)
    score_frozen=freeze_dir/"20_SCORE_SHEET_RATER1_20x6_FROZEN.tsv"
    qc_frozen=freeze_dir/"21_RATER1_BLINDED_SCORING_QC_FROZEN.json"
    shutil.copy2(score_src, score_frozen)
    shutil.copy2(qc_src, qc_frozen)

    freeze_manifest={
        "status":"RATER1_BLINDED_SCORING_FROZEN_AX06_AX15_V1",
        "freeze_date":"2026-08-29",
        "responses":20,
        "blind_ids":"AXR011-AXR030",
        "condition_key_used_before_freeze":False,
        "score_sheet_sha256":sha256(score_frozen),
        "qc_sha256":sha256(qc_frozen),
        "no_model_regeneration":True,
    }
    (freeze_dir/"90_FROZEN_MANIFEST.json").write_text(
        json.dumps(freeze_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (freeze_dir/"SHA256SUMS.txt").open("w",encoding="utf-8") as f:
        for p in sorted(freeze_dir.iterdir()):
            if p.is_file() and p.name!="SHA256SUMS.txt":
                f.write(f"{sha256(p)}  {p.name}\n")

    # Only now read the secret key.
    key_rows=read_jsonl(secret)
    if len(key_rows)!=20:
        raise SystemExit(f"ERROR secret key rows={len(key_rows)}, expected 20")
    key={r["blind_id"]:r for r in key_rows}
    if set(key)!=set(got_ids):
        raise SystemExit("ERROR secret key blind IDs do not match frozen score IDs")

    unblinded=[]
    for s in scores:
        k=key[s["blind_id"]]
        row={
            "blind_id":s["blind_id"],
            "benchmark_id":k["benchmark_id"],
            "condition":k["condition"],
        }
        for m in METRICS:
            row[m]=as_float(s[m])
        row["reviewer_notes"]=s.get("reviewer_notes","")
        row["review_status"]=s.get("review_status","")
        unblinded.append(row)

    unblind_dir.mkdir(parents=True)
    unblind_fields=["blind_id","benchmark_id","condition"]+METRICS+["reviewer_notes","review_status"]
    write_tsv(unblind_dir/"10_RATER1_UNBLINDED_SCORES_20.tsv", unblinded, unblind_fields)

    with (unblind_dir/"11_RATER1_UNBLINDED_SCORES_20.jsonl").open("w",encoding="utf-8") as f:
        for r in unblinded:
            f.write(json.dumps(r,ensure_ascii=False)+"\n")

    # Condition means
    means=[]
    for cond in ["FLAT_KG","MULTIBRANCH_KG"]:
        sub=[r for r in unblinded if r["condition"]==cond]
        if len(sub)!=10:
            raise SystemExit(f"ERROR {cond} rows={len(sub)} expected 10")
        row={"condition":cond,"n":len(sub)}
        for m in METRICS:
            row[m]=sum(r[m] for r in sub)/len(sub)
        means.append(row)
    write_tsv(
        unblind_dir/"20_FLAT_MULTI_MEANS_AX06_AX15.tsv",
        means,
        ["condition","n"]+METRICS
    )

    # Matched pairs and Multi - Flat differences
    by_bench={}
    for r in unblinded:
        by_bench.setdefault(r["benchmark_id"],{})[r["condition"]]=r

    pair_rows=[]
    for bid in sorted(by_bench):
        pair=by_bench[bid]
        if set(pair)!={"FLAT_KG","MULTIBRANCH_KG"}:
            raise SystemExit(f"ERROR incomplete pair for {bid}: {set(pair)}")
        flat=pair["FLAT_KG"]
        multi=pair["MULTIBRANCH_KG"]
        row={
            "benchmark_id":bid,
            "flat_blind_id":flat["blind_id"],
            "multi_blind_id":multi["blind_id"],
        }
        for m in METRICS:
            row[f"flat_{m}"]=flat[m]
            row[f"multi_{m}"]=multi[m]
            row[f"delta_multi_minus_flat_{m}"]=multi[m]-flat[m]
        pair_rows.append(row)

    pair_fields=["benchmark_id","flat_blind_id","multi_blind_id"]
    for m in METRICS:
        pair_fields += [f"flat_{m}",f"multi_{m}",f"delta_multi_minus_flat_{m}"]
    write_tsv(unblind_dir/"21_MATCHED_PAIRS_AX06_AX15.tsv", pair_rows, pair_fields)

    # Direction counts for each metric.
    directions=[]
    # For unsupported/collapse, lower is better. Others higher is better.
    lower_better={"collapse","unsupported"}
    for m in METRICS:
        deltas=[r[f"delta_multi_minus_flat_{m}"] for r in pair_rows]
        if m in lower_better:
            multi_better=sum(d<0 for d in deltas)
            flat_better=sum(d>0 for d in deltas)
        else:
            multi_better=sum(d>0 for d in deltas)
            flat_better=sum(d<0 for d in deltas)
        ties=sum(d==0 for d in deltas)
        directions.append({
            "metric":m,
            "multi_better":multi_better,
            "tie":ties,
            "flat_better":flat_better,
            "mean_delta_multi_minus_flat":sum(deltas)/len(deltas)
        })
    write_tsv(
        unblind_dir/"22_MATCHED_DIRECTION_COUNTS_AX06_AX15.tsv",
        directions,
        ["metric","multi_better","tie","flat_better","mean_delta_multi_minus_flat"]
    )

    summary={
        "status":"AX06_AX15_RATER1_UNBLINDED_RESULTS_FROZEN_V1",
        "freeze_date":"2026-08-29",
        "n_questions":10,
        "n_responses":20,
        "conditions":["FLAT_KG","MULTIBRANCH_KG"],
        "contrast":"MULTIBRANCH_KG minus FLAT_KG",
        "primary_metrics":["evidence_attribution","relation_awareness","uncertainty"],
        "secondary_metrics":["branch_coverage","collapse","unsupported"],
        "blind_scores_frozen_before_unblinding":True,
        "model_outputs_regenerated":False,
        "condition_means":means,
        "direction_counts":directions,
        "interpretation_note":"AX06-AX15 are the protocol-preserving extension cohort; do not describe AX01-AX15 as one initial preregistered set."
    }
    (unblind_dir/"30_AX06_AX15_UNBLINDED_SUMMARY.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"
    )

    with (unblind_dir/"SHA256SUMS.txt").open("w",encoding="utf-8") as f:
        for p in sorted(unblind_dir.iterdir()):
            if p.is_file() and p.name!="SHA256SUMS.txt":
                f.write(f"{sha256(p)}  {p.name}\n")

    print("SUCCESS")
    print("BLIND_SCORES_FROZEN_BEFORE_UNBLIND=True")
    print("UNBLINDED_RESPONSES=20")
    print("MATCHED_QUESTIONS=10")
    for row in means:
        print(row["condition"], "REL=", round(row["relation_awareness"],4),
              "EVID=", round(row["evidence_attribution"],4),
              "UNC=", round(row["uncertainty"],4))
    print("UNBLIND_DIR=",unblind_dir)
    print("GPU_REQUIRED=False")

if __name__=="__main__":
    main()

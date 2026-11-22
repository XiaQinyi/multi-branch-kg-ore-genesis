#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B10 historical reference/control extraction + paired analysis + final 50-response enhanced core.

CPU only.
- DOES NOT run any model.
- DOES NOT modify Main Experiment I / historical frozen files.
- DOES NOT modify A15 frozen results.
- Reads frozen files and writes a new derived-analysis directory only.

Authoritative B10 set is hard-locked below.
"""

from pathlib import Path
import argparse, csv, json, hashlib, random, math, sys

B10 = [
    "B01_YIFENG-FENGXIN_FORMATION_AGE",
    "B02_BAISHIHUASHAN_FORMATION_AGE",
    "B03_BAISHIHUASHAN_METAL_ENRICHMENT_MECHANISM",
    "B04_BAISHIHUASHAN_HYDROTHERMAL_ALTERATION_PROCESS",
    "B05_YIFENG-FENGXIN_MAGMATIC_DIFFERENTIATION",
    "B06_YIFENG-FENGXIN_HYDROTHERMAL_ALTERATION_PROCESS",
    "B07_YIFENG-FENGXIN_GEOCHEMICAL_SIGNATURE",
    "B08_YIFENG-FENGXIN_MINERAL_CHEMISTRY_OR_ASSEMBLAGE",
    "B09_BAISHIHUASHAN_MINERAL_CHEMISTRY_OR_ASSEMBLAGE",
    "B10_WEILASITU_MAGMA_SOURCE",
]
B10_SET = set(B10)

# Historical frozen sources
B_FLAT_REL = Path("runs/multibranch_kg_v2_grounded/16B_ExperimentII_Formal_Generation_v1/10_GROUNDED_FLAT_OUTPUTS.jsonl")
B_MULTI_REL = Path("runs/multibranch_kg_v2_grounded/16B_ExperimentII_Formal_Generation_v1/11_GROUNDED_MULTI_OUTPUTS.jsonl")
B_SCORE_REL = Path("runs/multibranch_kg_v2_grounded/18_ExperimentI_PerQuestion_Scores_Frozen_v1/10_PER_QUESTION_UNBLINDED_SCORES_52.tsv")

# A15 frozen sources
A_OLD_FLAT_REL = Path("runs/multibranch_kg_v2_grounded/20_AX_Prospective_Replication_v1/01_FROZEN_OUTPUTS/10_FLAT_KG_OUTPUTS_5.jsonl")
A_OLD_MULTI_REL = Path("runs/multibranch_kg_v2_grounded/20_AX_Prospective_Replication_v1/01_FROZEN_OUTPUTS/11_MULTIBRANCH_KG_OUTPUTS_5.jsonl")
A_NEW_FLAT_REL = Path("runs/multibranch_kg_v2_grounded/35_FinalEnhanced_AX06_AX15_Prospective_Extension_v1/01_FROZEN_OUTPUTS/10_FLAT_KG_OUTPUTS_10.jsonl")
A_NEW_MULTI_REL = Path("runs/multibranch_kg_v2_grounded/35_FinalEnhanced_AX06_AX15_Prospective_Extension_v1/01_FROZEN_OUTPUTS/11_MULTIBRANCH_KG_OUTPUTS_10.jsonl")
A_OLD_SCORE_REL = Path("runs/multibranch_kg_v2_grounded/20_AX_Prospective_Replication_v1/05_UNBLINDED_RATER1_RESULTS_FROZEN/10_RATER1_UNBLINDED_SCORES_10.tsv")
A_NEW_SCORE_REL = Path("runs/multibranch_kg_v2_grounded/35_FinalEnhanced_AX06_AX15_Prospective_Extension_v1/05_UNBLINDED_RATER1_RESULTS_FROZEN/10_RATER1_UNBLINDED_SCORES_20.tsv")
A15_ANALYSIS_REL = Path("runs/multibranch_kg_v2_grounded/36_Final_A15_Primary_Rater1_PairedAnalysis_v1/21_A15_PAIRED_EFFECTS_BOOTSTRAP_CI.tsv")

OUT_REL = Path("runs/multibranch_kg_v2_grounded/38_FinalEnhanced_B10_and_50Response_Core_v1")

METRICS = [
    "branch_coverage",
    "collapse",
    "relation_awareness",
    "unsupported",
    "uncertainty",
    "evidence_attribution",
]
LOWER_BETTER = {"collapse", "unsupported"}

ALIASES = {
    "branch_coverage": ["branch_coverage", "Branch Coverage"],
    "collapse": ["collapse", "single_answer_collapse", "Single-answer Collapse", "single-answer collapse"],
    "relation_awareness": ["relation_awareness", "Relation Awareness"],
    "unsupported": ["unsupported", "unsupported_claim_rate", "Unsupported Claim Rate"],
    "uncertainty": ["uncertainty", "uncertainty_quality", "Uncertainty Quality"],
    "evidence_attribution": ["evidence_attribution", "evidence_attribution_accuracy", "Evidence Attribution Accuracy"],
}
ID_ALIASES = ["benchmark_id", "question_id", "id", "benchmark"]
COND_ALIASES = ["condition", "Condition", "arm", "method"]

BOOT_N = 20000
BOOT_SEED = 20260829

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

def write_jsonl(p, rows):
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def sha256(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def first_value(row, aliases):
    for a in aliases:
        if a in row and row[a] not in (None, ""):
            return row[a]
    return None

def get_id(row):
    v=first_value(row, ID_ALIASES)
    if v is not None:
        return str(v).strip()
    # Last-resort exact-token detection from row values, without fuzzy matching.
    vals={str(v).strip() for v in row.values() if v is not None}
    for bid in B10:
        if bid in vals:
            return bid
    return None

def normalize_condition(v):
    if v is None:
        return None
    s=str(v).strip().upper().replace("-", "_").replace(" ", "_")
    if "MULTI" in s:
        return "MULTIBRANCH_KG"
    if "FLAT" in s:
        return "FLAT_KG"
    return None

def metric_value(row, metric):
    v=first_value(row, ALIASES[metric])
    if v is None or str(v).strip()=="":
        raise ValueError(f"missing metric={metric}")
    s=str(v).strip()
    if s.upper() in {"NA","N/A","NONE","NULL"}:
        raise ValueError(f"N/A metric={metric}")
    return float(s)

def raw_output_value(row):
    for k in ["raw_output","model_response","response","output","answer_raw"]:
        if k in row:
            return row[k]
    return None

def percentile(xs,q):
    xs=sorted(xs)
    pos=(len(xs)-1)*q
    lo=int(math.floor(pos)); hi=int(math.ceil(pos))
    if lo==hi:
        return xs[lo]
    w=pos-lo
    return xs[lo]*(1-w)+xs[hi]*w

def bootstrap_ci(ds, seed):
    rng=random.Random(seed)
    n=len(ds)
    boots=[]
    for _ in range(BOOT_N):
        boots.append(sum(ds[rng.randrange(n)] for _ in range(n))/n)
    return percentile(boots,0.025), percentile(boots,0.975)

def exact_sign_p(ds):
    nz=[x for x in ds if x != 0]
    n=len(nz)
    if n==0:
        return 1.0
    pos=sum(x>0 for x in nz)
    k=min(pos,n-pos)
    tail=sum(math.comb(n,i) for i in range(k+1))/(2**n)
    return min(1.0,2*tail)

def paired_effects(pair_rows, cohort_name):
    rows=[]
    for i,m in enumerate(METRICS):
        ds=[r[f"delta_multi_minus_flat_{m}"] for r in pair_rows]
        lo,hi=bootstrap_ci(ds, BOOT_SEED+i+(0 if cohort_name=="B10_REFERENCE" else 100))
        flat=[r[f"flat_{m}"] for r in pair_rows]
        multi=[r[f"multi_{m}"] for r in pair_rows]
        if m in LOWER_BETTER:
            mb=sum(d<0 for d in ds); fb=sum(d>0 for d in ds)
        else:
            mb=sum(d>0 for d in ds); fb=sum(d<0 for d in ds)
        rows.append({
            "cohort":cohort_name,
            "metric":m,
            "n":len(ds),
            "flat_mean":sum(flat)/len(flat),
            "multi_mean":sum(multi)/len(multi),
            "mean_delta_multi_minus_flat":sum(ds)/len(ds),
            "bootstrap95_low":lo,
            "bootstrap95_high":hi,
            "multi_better":mb,
            "tie":sum(d==0 for d in ds),
            "flat_better":fb,
            "exact_sign_p_two_sided_on_nonzero_deltas":exact_sign_p(ds),
            "better_direction":"LOWER" if m in LOWER_BETTER else "HIGHER",
        })
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".")
    args=ap.parse_args()
    root=Path(args.root).resolve()

    srcs = [
        B_FLAT_REL,B_MULTI_REL,B_SCORE_REL,
        A_OLD_FLAT_REL,A_OLD_MULTI_REL,A_NEW_FLAT_REL,A_NEW_MULTI_REL,
        A_OLD_SCORE_REL,A_NEW_SCORE_REL,A15_ANALYSIS_REL
    ]
    missing=[root/p for p in srcs if not (root/p).exists()]
    if missing:
        print("ERROR missing required frozen source(s):")
        for p in missing: print(" -",p)
        sys.exit(2)

    out=root/OUT_REL
    if out.exists():
        raise SystemExit(f"ERROR refusing overwrite existing output dir: {out}")
    out.mkdir(parents=True)

    # ---------- 1. Extract B10 frozen Grounded Flat/Multi outputs ----------
    b_outputs=[]
    for source_rel,cond in [(B_FLAT_REL,"FLAT_KG"),(B_MULTI_REL,"MULTIBRANCH_KG")]:
        rows=read_jsonl(root/source_rel)
        selected=[]
        for r in rows:
            bid=get_id(r)
            if bid in B10_SET:
                rr={
                    "cohort":"B10_REFERENCE",
                    "benchmark_id":bid,
                    "condition":cond,
                    "historical_condition":first_value(r,COND_ALIASES),
                    "raw_output":raw_output_value(r),
                    "source_file":str(source_rel),
                }
                # Preserve full original record too, for auditability.
                rr["original_record"]=r
                selected.append(rr)
        if len(selected)!=10:
            raise SystemExit(f"ERROR {source_rel}: selected B10 rows={len(selected)}, expected 10")
        if set(r["benchmark_id"] for r in selected)!=B10_SET:
            raise SystemExit(f"ERROR {source_rel}: B10 ID set mismatch")
        b_outputs.extend(selected)

    if len({(r["benchmark_id"],r["condition"]) for r in b_outputs})!=20:
        raise SystemExit("ERROR B10 output cells are not exactly 20 unique question-condition cells")

    write_jsonl(out/"10_B10_FROZEN_FLAT_MULTI_OUTPUTS_20.jsonl", b_outputs)

    # ---------- 2. Extract existing historical B10 scores ----------
    score52=read_tsv(root/B_SCORE_REL)
    b_scores=[]
    skipped_b10_other_conditions=0

    for r in score52:
        bid=get_id(r)
        if bid not in B10_SET:
            continue
        cond_raw=first_value(r,COND_ALIASES)
        cond=normalize_condition(cond_raw)
        if cond not in {"FLAT_KG","MULTIBRANCH_KG"}:
            skipped_b10_other_conditions += 1
            continue
        try:
            rr={
                "cohort":"B10_REFERENCE",
                "benchmark_id":bid,
                "condition":cond,
                "historical_condition":cond_raw,
            }
            for m in METRICS:
                rr[m]=metric_value(r,m)
        except Exception as e:
            raise SystemExit(f"ERROR score parse for {bid} condition={cond_raw}: {e}\nROW={r}")
        b_scores.append(rr)

    if len(b_scores)!=20:
        print("ERROR expected exactly 20 B10 Flat/Multi score rows, got",len(b_scores))
        print("B10 other-condition rows skipped =",skipped_b10_other_conditions)
        print("Detected B10 score rows:")
        for r in b_scores:
            print(r["benchmark_id"],r["historical_condition"],r["condition"])
        sys.exit(3)

    keys={(r["benchmark_id"],r["condition"]) for r in b_scores}
    expected={(bid,c) for bid in B10 for c in ["FLAT_KG","MULTIBRANCH_KG"]}
    if keys!=expected:
        print("ERROR B10 score cell mismatch")
        print("missing=",sorted(expected-keys))
        print("extra=",sorted(keys-expected))
        sys.exit(4)

    write_tsv(
        out/"11_B10_EXISTING_RATER1_SCORES_20.tsv",
        b_scores,
        ["cohort","benchmark_id","condition","historical_condition"]+METRICS
    )

    # ---------- 3. B10 paired analysis ----------
    by_b={}
    for r in b_scores:
        by_b.setdefault(r["benchmark_id"],{})[r["condition"]]=r

    b_pairs=[]
    for bid in B10:
        d=by_b.get(bid,{})
        if set(d)!={"FLAT_KG","MULTIBRANCH_KG"}:
            raise SystemExit(f"ERROR incomplete B10 pair {bid}")
        flat=d["FLAT_KG"]; multi=d["MULTIBRANCH_KG"]
        row={"cohort":"B10_REFERENCE","benchmark_id":bid}
        for m in METRICS:
            row[f"flat_{m}"]=flat[m]
            row[f"multi_{m}"]=multi[m]
            row[f"delta_multi_minus_flat_{m}"]=multi[m]-flat[m]
        b_pairs.append(row)

    pair_fields=["cohort","benchmark_id"]
    for m in METRICS:
        pair_fields += [f"flat_{m}",f"multi_{m}",f"delta_multi_minus_flat_{m}"]
    write_tsv(out/"20_B10_MATCHED_PAIRS_10.tsv",b_pairs,pair_fields)

    b_effects=paired_effects(b_pairs,"B10_REFERENCE")
    eff_fields=["cohort","metric","n","flat_mean","multi_mean","mean_delta_multi_minus_flat",
                "bootstrap95_low","bootstrap95_high","multi_better","tie","flat_better",
                "exact_sign_p_two_sided_on_nonzero_deltas","better_direction"]
    write_tsv(out/"21_B10_PAIRED_EFFECTS_BOOTSTRAP_CI.tsv",b_effects,eff_fields)

    # ---------- 4. Assemble final 50-response core ----------
    # A15 outputs
    a_outputs=[]
    for source_rel,cond,prov in [
        (A_OLD_FLAT_REL,"FLAT_KG","A15_INITIAL_AX01_AX05"),
        (A_OLD_MULTI_REL,"MULTIBRANCH_KG","A15_INITIAL_AX01_AX05"),
        (A_NEW_FLAT_REL,"FLAT_KG","A15_EXTENSION_AX06_AX15"),
        (A_NEW_MULTI_REL,"MULTIBRANCH_KG","A15_EXTENSION_AX06_AX15"),
    ]:
        for r in read_jsonl(root/source_rel):
            bid=get_id(r)
            if not bid:
                raise SystemExit(f"ERROR cannot identify A15 output row in {source_rel}")
            a_outputs.append({
                "cohort":"A15_PRIMARY",
                "provenance":prov,
                "benchmark_id":bid,
                "condition":cond,
                "raw_output":raw_output_value(r),
                "source_file":str(source_rel),
                "original_record":r,
            })

    if len(a_outputs)!=30:
        raise SystemExit(f"ERROR A15 output rows={len(a_outputs)}, expected 30")
    if len({(r["benchmark_id"],r["condition"]) for r in a_outputs})!=30:
        raise SystemExit("ERROR A15 output cells not unique 30")

    combined_outputs = a_outputs + b_outputs
    if len(combined_outputs)!=50:
        raise SystemExit("ERROR final outputs !=50")
    write_jsonl(out/"30_FINAL_ENHANCED_CORE_FROZEN_OUTPUTS_50.jsonl",combined_outputs)

    # A15 scores, already frozen/unblinded.
    a_scores=[]
    for source_rel,prov in [
        (A_OLD_SCORE_REL,"A15_INITIAL_AX01_AX05"),
        (A_NEW_SCORE_REL,"A15_EXTENSION_AX06_AX15"),
    ]:
        for r in read_tsv(root/source_rel):
            bid=get_id(r)
            cond=normalize_condition(first_value(r,COND_ALIASES))
            if not bid or cond not in {"FLAT_KG","MULTIBRANCH_KG"}:
                raise SystemExit(f"ERROR A15 score id/condition parse in {source_rel}: {r}")
            rr={
                "cohort":"A15_PRIMARY",
                "provenance":prov,
                "benchmark_id":bid,
                "condition":cond,
                "historical_condition":first_value(r,COND_ALIASES),
            }
            for m in METRICS:
                rr[m]=metric_value(r,m)
            a_scores.append(rr)
    if len(a_scores)!=30:
        raise SystemExit(f"ERROR A15 score rows={len(a_scores)}, expected 30")

    combined_scores = a_scores + [
        dict(r, provenance="HISTORICAL_B10_REFERENCE") for r in b_scores
    ]
    if len(combined_scores)!=50:
        raise SystemExit("ERROR final score rows !=50")

    write_tsv(
        out/"31_FINAL_ENHANCED_CORE_RATER1_SCORES_50.tsv",
        combined_scores,
        ["cohort","provenance","benchmark_id","condition","historical_condition"]+METRICS
    )

    # Combined 25 matched pairs.
    by_all={}
    for r in combined_scores:
        key=(r["cohort"],r["benchmark_id"])
        by_all.setdefault(key,{})[r["condition"]]=r

    if len(by_all)!=25:
        raise SystemExit(f"ERROR final unique question count={len(by_all)}, expected 25")

    all_pairs=[]
    for (cohort,bid),d in sorted(by_all.items()):
        if set(d)!={"FLAT_KG","MULTIBRANCH_KG"}:
            raise SystemExit(f"ERROR final incomplete pair {cohort} {bid}")
        flat=d["FLAT_KG"]; multi=d["MULTIBRANCH_KG"]
        row={"cohort":cohort,"benchmark_id":bid}
        for m in METRICS:
            row[f"flat_{m}"]=flat[m]
            row[f"multi_{m}"]=multi[m]
            row[f"delta_multi_minus_flat_{m}"]=multi[m]-flat[m]
        all_pairs.append(row)
    write_tsv(out/"32_FINAL_ENHANCED_CORE_MATCHED_PAIRS_25.tsv",all_pairs,pair_fields)

    a_pairs=[r for r in all_pairs if r["cohort"]=="A15_PRIMARY"]
    bb_pairs=[r for r in all_pairs if r["cohort"]=="B10_REFERENCE"]
    if len(a_pairs)!=15 or len(bb_pairs)!=10:
        raise SystemExit(f"ERROR pair strata A={len(a_pairs)} B={len(bb_pairs)} expected 15/10")

    # A15 independently recomputed from frozen scores, then cross-check against existing analysis.
    a_effects=paired_effects(a_pairs,"A15_PRIMARY")
    omnibus=paired_effects(all_pairs,"ALL25_DESCRIPTIVE_ONLY")
    stratified=a_effects+b_effects+omnibus
    write_tsv(out/"33_FINAL_ENHANCED_CORE_STRATIFIED_AND_OMNIBUS_EFFECTS.tsv",stratified,eff_fields)

    # Cross-check A15 values against previously frozen A15 result table.
    prev=read_tsv(root/A15_ANALYSIS_REL)
    prev_by={r["metric"]:r for r in prev}
    a15_crosscheck=[]
    for r in a_effects:
        m=r["metric"]
        pr=prev_by.get(m)
        if pr is None:
            raise SystemExit(f"ERROR previous A15 analysis missing metric {m}")
        checks={
            "flat_mean": abs(float(pr["flat_mean"])-r["flat_mean"])<1e-9,
            "multi_mean": abs(float(pr["multi_mean"])-r["multi_mean"])<1e-9,
            "mean_delta": abs(float(pr["mean_delta_multi_minus_flat"])-r["mean_delta_multi_minus_flat"])<1e-9,
        }
        a15_crosscheck.append({"metric":m,**checks,"all_match":all(checks.values())})
    if not all(x["all_match"] for x in a15_crosscheck):
        raise SystemExit(f"ERROR A15 recomputation does not match frozen analysis: {a15_crosscheck}")
    write_tsv(out/"34_A15_FROZEN_RESULT_CROSSCHECK.tsv",a15_crosscheck,
              ["metric","flat_mean","multi_mean","mean_delta","all_match"])

    # Compact final core index.
    index=[]
    for r in combined_outputs:
        index.append({
            "cohort":r["cohort"],
            "provenance":r.get("provenance","HISTORICAL_B10_REFERENCE"),
            "benchmark_id":r["benchmark_id"],
            "condition":r["condition"],
            "source_file":r["source_file"],
        })
    write_tsv(out/"35_FINAL_ENHANCED_CORE_RESPONSE_INDEX_50.tsv",index,
              ["cohort","provenance","benchmark_id","condition","source_file"])

    manifest={
        "status":"FINAL_ENHANCED_CORE_25Q_50RESPONSES_V1",
        "date":"2026-08-29",
        "gpu_required":False,
        "model_generation_performed":False,
        "historical_files_modified":False,
        "A15":{
            "role":"PRIMARY_SCIENTIFIC_COHORT",
            "questions":15,
            "responses":30,
            "timing_provenance":"AX01-AX05 initial prospective strict set; AX06-AX15 protocol-preserving extension frozen before outputs for added questions."
        },
        "B10":{
            "role":"HISTORICAL_REFERENCE_CONTROL",
            "questions":10,
            "responses":20,
            "ids":B10,
            "outputs_source_flat":str(B_FLAT_REL),
            "outputs_source_multi":str(B_MULTI_REL),
            "scores_source":str(B_SCORE_REL),
        },
        "final_core":{"questions":25,"responses":50,"conditions":["FLAT_KG","MULTIBRANCH_KG"]},
        "reporting_guardrails":[
            "Primary inferential emphasis remains A15.",
            "B10 is a historical reference/control cohort.",
            "ALL25 omnibus results are descriptive only.",
            "Do not interpret A15-vs-B10 differences as a pure causal relation-type interaction because cohort construction histories differ.",
            "No unfavorable metric is dropped."
        ],
        "metrics":METRICS,
        "bootstrap_replicates":BOOT_N,
        "bootstrap_seed":BOOT_SEED,
        "source_sha256":{str(p):sha256(root/p) for p in srcs},
    }
    (out/"90_FROZEN_MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")

    with (out/"SHA256SUMS.txt").open("w",encoding="utf-8") as f:
        for p in sorted(out.iterdir()):
            if p.is_file() and p.name!="SHA256SUMS.txt":
                f.write(f"{sha256(p)}  {p.name}\n")

    print("SUCCESS")
    print("B10_FROZEN_OUTPUTS=20")
    print("B10_EXISTING_SCORES=20")
    print("B10_MATCHED_QUESTIONS=10")
    print("A15_PRIMARY_QUESTIONS=15")
    print("FINAL_CORE_QUESTIONS=25")
    print("FINAL_CORE_RESPONSES=50")
    print("A15_FROZEN_RESULT_CROSSCHECK=True")
    print("")
    print("=== B10 PAIRED EFFECTS ===")
    for r in b_effects:
        print(r["metric"],
              "flat=",round(r["flat_mean"],4),
              "multi=",round(r["multi_mean"],4),
              "delta=",round(r["mean_delta_multi_minus_flat"],4),
              "CI=[",round(r["bootstrap95_low"],4),",",round(r["bootstrap95_high"],4),"]",
              "dir=",f"{r['multi_better']}/{r['tie']}/{r['flat_better']}")
    print("")
    print("=== FINAL 25Q DESCRIPTIVE OMNIBUS ===")
    for r in omnibus:
        print(r["metric"],
              "flat=",round(r["flat_mean"],4),
              "multi=",round(r["multi_mean"],4),
              "delta=",round(r["mean_delta_multi_minus_flat"],4),
              "CI=[",round(r["bootstrap95_low"],4),",",round(r["bootstrap95_high"],4),"]")
    print("")
    print("OUTPUT_DIR=",out)
    print("GPU_REQUIRED=False")

if __name__=="__main__":
    main()

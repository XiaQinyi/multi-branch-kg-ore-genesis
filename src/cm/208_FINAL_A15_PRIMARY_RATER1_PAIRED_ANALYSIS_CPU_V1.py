#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combine frozen Rater-1 AX01-AX05 and AX06-AX15 results into final A15 primary cohort.
CPU only. No model generation. No score modification.

Reads:
- runs/.../20_AX_Prospective_Replication_v1/05_UNBLINDED_RATER1_RESULTS_FROZEN/10_RATER1_UNBLINDED_SCORES_10.tsv
- runs/.../35_FinalEnhanced_AX06_AX15_Prospective_Extension_v1/05_UNBLINDED_RATER1_RESULTS_FROZEN/10_RATER1_UNBLINDED_SCORES_20.tsv

Writes:
- runs/.../36_Final_A15_Primary_Rater1_PairedAnalysis_v1/
"""

from pathlib import Path
import argparse, csv, json, hashlib, random, math, sys

OLD_REL = Path("runs/multibranch_kg_v2_grounded/20_AX_Prospective_Replication_v1/05_UNBLINDED_RATER1_RESULTS_FROZEN/10_RATER1_UNBLINDED_SCORES_10.tsv")
NEW_REL = Path("runs/multibranch_kg_v2_grounded/35_FinalEnhanced_AX06_AX15_Prospective_Extension_v1/05_UNBLINDED_RATER1_RESULTS_FROZEN/10_RATER1_UNBLINDED_SCORES_20.tsv")
OUT_REL = Path("runs/multibranch_kg_v2_grounded/36_Final_A15_Primary_Rater1_PairedAnalysis_v1")

METRICS = [
    "branch_coverage",
    "collapse",
    "relation_awareness",
    "unsupported",
    "uncertainty",
    "evidence_attribution",
]
LOWER_BETTER = {"collapse","unsupported"}
BOOT_SEED = 20260829
BOOT_N = 20000

def read_tsv(p):
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))

def write_tsv(p, rows, fields):
    with p.open("w", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter="\t",extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def sha256(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def f(v):
    return float(v)

def percentile(xs, q):
    xs=sorted(xs)
    if not xs:
        return None
    pos=(len(xs)-1)*q
    lo=int(math.floor(pos)); hi=int(math.ceil(pos))
    if lo==hi:
        return xs[lo]
    w=pos-lo
    return xs[lo]*(1-w)+xs[hi]*w

def bootstrap_mean_ci(deltas, seed):
    rng=random.Random(seed)
    n=len(deltas)
    boots=[]
    for _ in range(BOOT_N):
        sample=[deltas[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample)/n)
    return percentile(boots,0.025), percentile(boots,0.975)

def exact_sign_p(deltas):
    nz=[d for d in deltas if d != 0]
    n=len(nz)
    if n==0:
        return 1.0
    pos=sum(d>0 for d in nz)
    k=min(pos,n-pos)
    # exact two-sided sign test
    tail=sum(math.comb(n,i) for i in range(0,k+1))/(2**n)
    return min(1.0, 2*tail)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".")
    args=ap.parse_args()
    root=Path(args.root).resolve()
    old=root/OLD_REL
    new=root/NEW_REL
    out=root/OUT_REL

    for p in [old,new]:
        if not p.exists():
            raise SystemExit(f"ERROR missing: {p}")
    if out.exists():
        raise SystemExit(f"ERROR refusing overwrite: {out}")
    out.mkdir(parents=True)

    old_rows=read_tsv(old)
    new_rows=read_tsv(new)
    if len(old_rows)!=10 or len(new_rows)!=20:
        raise SystemExit(f"ERROR expected old=10,new=20; got {len(old_rows)},{len(new_rows)}")

    all_rows=[]
    for cohort,rows in [("INITIAL_AX01_AX05",old_rows),("EXTENSION_AX06_AX15",new_rows)]:
        for r in rows:
            rr=dict(r)
            rr["cohort_provenance"]=cohort
            for m in METRICS:
                rr[m]=f(rr[m])
            all_rows.append(rr)

    # 15 unique questions x 2 conditions
    by_bench={}
    for r in all_rows:
        bid=r["benchmark_id"]
        cond=r["condition"]
        by_bench.setdefault(bid,{})
        if cond in by_bench[bid]:
            raise SystemExit(f"ERROR duplicate {bid} {cond}")
        by_bench[bid][cond]=r
    if len(by_bench)!=15:
        raise SystemExit(f"ERROR unique questions={len(by_bench)}, expected 15")

    pairs=[]
    for bid in sorted(by_bench):
        d=by_bench[bid]
        if set(d)!={"FLAT_KG","MULTIBRANCH_KG"}:
            raise SystemExit(f"ERROR incomplete pair {bid}: {set(d)}")
        flat=d["FLAT_KG"]; multi=d["MULTIBRANCH_KG"]
        prov=flat["cohort_provenance"]
        if multi["cohort_provenance"]!=prov:
            raise SystemExit(f"ERROR provenance mismatch {bid}")
        row={"benchmark_id":bid,"cohort_provenance":prov,
             "flat_blind_id":flat.get("blind_id",""),
             "multi_blind_id":multi.get("blind_id","")}
        for m in METRICS:
            row[f"flat_{m}"]=flat[m]
            row[f"multi_{m}"]=multi[m]
            row[f"delta_multi_minus_flat_{m}"]=multi[m]-flat[m]
        pairs.append(row)

    pair_fields=["benchmark_id","cohort_provenance","flat_blind_id","multi_blind_id"]
    for m in METRICS:
        pair_fields += [f"flat_{m}",f"multi_{m}",f"delta_multi_minus_flat_{m}"]
    write_tsv(out/"10_A15_MATCHED_PAIRS_15.tsv",pairs,pair_fields)

    # Means
    means=[]
    for cond in ["FLAT_KG","MULTIBRANCH_KG"]:
        sub=[r for r in all_rows if r["condition"]==cond]
        row={"condition":cond,"n":len(sub)}
        for m in METRICS:
            row[m]=sum(r[m] for r in sub)/len(sub)
        means.append(row)
    write_tsv(out/"20_A15_CONDITION_MEANS.tsv",means,["condition","n"]+METRICS)

    # Paired effects + bootstrap CIs + sign test sensitivity
    eff=[]
    for idx,m in enumerate(METRICS):
        ds=[r[f"delta_multi_minus_flat_{m}"] for r in pairs]
        lo,hi=bootstrap_mean_ci(ds,BOOT_SEED+idx)
        mean_d=sum(ds)/len(ds)
        if m in LOWER_BETTER:
            multi_better=sum(d<0 for d in ds)
            flat_better=sum(d>0 for d in ds)
        else:
            multi_better=sum(d>0 for d in ds)
            flat_better=sum(d<0 for d in ds)
        ties=sum(d==0 for d in ds)
        eff.append({
            "metric":m,
            "n":len(ds),
            "flat_mean":next(r[m] for r in means if r["condition"]=="FLAT_KG"),
            "multi_mean":next(r[m] for r in means if r["condition"]=="MULTIBRANCH_KG"),
            "mean_delta_multi_minus_flat":mean_d,
            "bootstrap95_low":lo,
            "bootstrap95_high":hi,
            "multi_better":multi_better,
            "tie":ties,
            "flat_better":flat_better,
            "exact_sign_p_two_sided_on_nonzero_deltas":exact_sign_p(ds),
            "better_direction":"LOWER" if m in LOWER_BETTER else "HIGHER",
        })
    eff_fields=["metric","n","flat_mean","multi_mean","mean_delta_multi_minus_flat",
                "bootstrap95_low","bootstrap95_high","multi_better","tie","flat_better",
                "exact_sign_p_two_sided_on_nonzero_deltas","better_direction"]
    write_tsv(out/"21_A15_PAIRED_EFFECTS_BOOTSTRAP_CI.tsv",eff,eff_fields)

    # Provenance-separated descriptive results for transparent reporting
    strata=[]
    for prov in ["INITIAL_AX01_AX05","EXTENSION_AX06_AX15"]:
        pp=[r for r in pairs if r["cohort_provenance"]==prov]
        for m in METRICS:
            ds=[r[f"delta_multi_minus_flat_{m}"] for r in pp]
            strata.append({
                "cohort_provenance":prov,
                "metric":m,
                "n":len(pp),
                "mean_delta_multi_minus_flat":sum(ds)/len(ds),
                "multi_better":sum((d<0 if m in LOWER_BETTER else d>0) for d in ds),
                "tie":sum(d==0 for d in ds),
                "flat_better":sum((d>0 if m in LOWER_BETTER else d<0) for d in ds),
            })
    write_tsv(out/"22_A15_RESULTS_BY_PROVENANCE_STRATUM.tsv",strata,
              ["cohort_provenance","metric","n","mean_delta_multi_minus_flat",
               "multi_better","tie","flat_better"])

    summary={
        "status":"FINAL_A15_PRIMARY_RATER1_PAIRED_ANALYSIS_V1",
        "date":"2026-08-29",
        "primary_cohort":"strict competing AX01-AX15",
        "n_questions":15,
        "n_responses":30,
        "contrast":"MULTIBRANCH_KG minus FLAT_KG",
        "primary_metrics":["evidence_attribution","relation_awareness","uncertainty"],
        "secondary_metrics":["branch_coverage","collapse","unsupported"],
        "timing_provenance":{
            "AX01_AX05":"initial prospective strict set; results observed before extension",
            "AX06_AX15":"protocol-preserving extension using unchanged A1-A5; frozen before outputs for added questions"
        },
        "reporting_guardrail":"May report final strict benchmark n=15, but Methods must disclose the 5+10 timing provenance; do not claim a single initial preregistered 15-question set.",
        "p_values":"secondary sensitivity only",
        "bootstrap_replicates":BOOT_N,
        "bootstrap_seed":BOOT_SEED,
        "gpu_required":False
    }
    (out/"90_FROZEN_MANIFEST.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    with (out/"SHA256SUMS.txt").open("w",encoding="utf-8") as fh:
        for p in sorted(out.iterdir()):
            if p.is_file() and p.name!="SHA256SUMS.txt":
                fh.write(f"{sha256(p)}  {p.name}\n")

    print("SUCCESS")
    print("A15_QUESTIONS=15")
    print("A15_RESPONSES=30")
    for r in eff:
        print(r["metric"],
              "flat=",round(r["flat_mean"],4),
              "multi=",round(r["multi_mean"],4),
              "delta=",round(r["mean_delta_multi_minus_flat"],4),
              "CI=[",round(r["bootstrap95_low"],4),",",round(r["bootstrap95_high"],4),"]",
              "dir=",f"{r['multi_better']}/{r['tie']}/{r['flat_better']}")
    print("OUTPUT_DIR=",out)
    print("GPU_REQUIRED=False")

if __name__=="__main__":
    main()

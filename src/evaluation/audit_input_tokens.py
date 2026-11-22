#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path

from transformers import AutoTokenizer
import transformers

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
args = ap.parse_args()

ROOT = Path("MultiBranch_LLM_Experiment_v1")
OUT = Path("Paper_Final_Evaluation_v1/01_Token_Audit")
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "LLM_ONLY": "llm_only.jsonl",
    "TEXT_RAG": "text_rag.jsonl",
    "FLAT_KG": "flat_kg.jsonl",
    "MULTIBRANCH_KG": "multibranch_kg.jsonl",
}

def load_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

print("Loading tokenizer:", args.model)
tok = AutoTokenizer.from_pretrained(
    args.model,
    trust_remote_code=True
)

records = []

for condition, fn in FILES.items():
    path = ROOT / "01_INPUTS" / fn
    rows = load_jsonl(path)

    assert len(rows) == 13, (condition, len(rows))

    for r in rows:
        msgs = [
            {"role": "system", "content": r["system_prompt"]},
            {"role": "user", "content": r["user_prompt"]},
        ]

        # Exact reproduction of frozen run_all_hf.py
        text = tok.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True
        )

        x = tok(text, return_tensors="pt")
        input_tokens = int(x["input_ids"].shape[1])

        context_tokens = len(
            tok(
                r.get("context", ""),
                add_special_tokens=False
            )["input_ids"]
        )

        records.append({
            "benchmark_id": r["benchmark_id"],
            "condition": condition,
            "input_tokens": input_tokens,
            "context_tokens": context_tokens,
            "context_chars": len(r.get("context", "")),
            "rendered_chars": len(text),
        })

# 1. Per-question counts
per_path = OUT / "token_counts_per_question.csv"
with per_path.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=records[0].keys())
    w.writeheader()
    w.writerows(records)

# 2. Summary
summaries = []

for condition in FILES:
    vals = [
        r["input_tokens"]
        for r in records
        if r["condition"] == condition
    ]
    ctx = [
        r["context_tokens"]
        for r in records
        if r["condition"] == condition
    ]

    summaries.append({
        "condition": condition,
        "n": len(vals),
        "mean_input_tokens": round(statistics.mean(vals), 2),
        "sd_input_tokens": round(statistics.stdev(vals), 2),
        "median_input_tokens": statistics.median(vals),
        "min_input_tokens": min(vals),
        "max_input_tokens": max(vals),
        "mean_context_tokens": round(statistics.mean(ctx), 2),
    })

sum_path = OUT / "token_summary_by_condition.csv"
with sum_path.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=summaries[0].keys())
    w.writeheader()
    w.writerows(summaries)

# 3. Strict Flat vs Multi knowledge-atom comparison
flat = {
    r["benchmark_id"]: r
    for r in load_jsonl(ROOT / "01_INPUTS/flat_kg.jsonl")
}
multi = {
    r["benchmark_id"]: r
    for r in load_jsonl(ROOT / "01_INPUTS/multibranch_kg.jsonl")
}

pat = re.compile(r"^\[(S\d+)\]\s*(.*)$")

def atoms(context):
    d = {}
    for line in context.splitlines():
        m = pat.match(line.strip())
        if m:
            d[m.group(1)] = m.group(2).strip()
    return d

atom_rows = []

for qid in sorted(flat):
    fa = atoms(flat[qid]["context"])
    ma = atoms(multi[qid]["context"])

    same_ids = set(fa) == set(ma)
    same_payload = same_ids and all(fa[k] == ma[k] for k in fa)

    atom_rows.append({
        "benchmark_id": qid,
        "flat_atoms": len(fa),
        "multi_atoms": len(ma),
        "same_atom_ids": same_ids,
        "same_atom_payload": same_payload,
    })

atom_path = OUT / "flat_vs_multi_atom_check.csv"
with atom_path.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=atom_rows[0].keys())
    w.writeheader()
    w.writerows(atom_rows)

# 4. Paired Flat/Multi token differences
lookup = {
    (r["benchmark_id"], r["condition"]): r["input_tokens"]
    for r in records
}

diff_rows = []

for qid in sorted(flat):
    f = lookup[(qid, "FLAT_KG")]
    m = lookup[(qid, "MULTIBRANCH_KG")]

    diff_rows.append({
        "benchmark_id": qid,
        "flat_input_tokens": f,
        "multibranch_input_tokens": m,
        "multi_minus_flat": m - f,
        "multi_to_flat_ratio": round(m / f, 4),
    })

diff_path = OUT / "flat_vs_multi_token_difference.csv"
with diff_path.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=diff_rows[0].keys())
    w.writeheader()
    w.writerows(diff_rows)

# 5. Manifest
manifest = {
    "status": "TOKEN_AUDIT_COMPLETE",
    "model_path": args.model,
    "tokenizer_name_or_path": str(tok.name_or_path),
    "tokenizer_class": tok.__class__.__name__,
    "transformers_version": transformers.__version__,
    "n_questions": 13,
    "n_conditions": 4,
    "n_total_inputs": len(records),
    "token_definition": (
        "system+user messages -> apply_chat_template("
        "tokenize=False, add_generation_prompt=True) -> "
        "tokenizer(text, return_tensors='pt')"
    ),
    "flat_multi_same_atom_ids":
        all(r["same_atom_ids"] for r in atom_rows),
    "flat_multi_same_atom_payload":
        all(r["same_atom_payload"] for r in atom_rows),
    "input_sha256": {
        fn: sha256(ROOT / "01_INPUTS" / fn)
        for fn in FILES.values()
    },
}

manifest_path = OUT / "TOKEN_AUDIT_MANIFEST.json"
manifest_path.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

(OUT / "TOKEN_AUDIT_COMPLETE.marker").write_text(
    "TOKEN_AUDIT_COMPLETE\n",
    encoding="utf-8"
)

print("\n" + "=" * 72)
print("TOKEN AUDIT SUMMARY")
print("=" * 72)

for s in summaries:
    print(
        f"{s['condition']:18s} "
        f"mean={s['mean_input_tokens']:8.2f} "
        f"sd={s['sd_input_tokens']:8.2f} "
        f"median={s['median_input_tokens']} "
        f"range={s['min_input_tokens']}-{s['max_input_tokens']}"
    )

print("\nFlat/Multi same atom IDs:",
      manifest["flat_multi_same_atom_ids"])
print("Flat/Multi same atom payload:",
      manifest["flat_multi_same_atom_payload"])

print(
    "Mean Multi-Flat token difference:",
    round(
        statistics.mean(
            r["multi_minus_flat"] for r in diff_rows
        ),
        2
    )
)

print("\nDONE")

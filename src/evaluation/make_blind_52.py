#!/usr/bin/env python3

import csv
import hashlib
import json
import random
import re
import shutil
from pathlib import Path

ROOT = Path("MultiBranch_LLM_Experiment_v1")
OUT = Path("Paper_Final_Evaluation_v1/02_Blind_Eval")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260808

CONDITIONS = {
    "LLM_ONLY": "llm_only",
    "TEXT_RAG": "text_rag",
    "FLAT_KG": "flat_kg",
    "MULTIBRANCH_KG": "multibranch_kg",
}

def load_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# --------------------------------------------------
# Load frozen Gold
# --------------------------------------------------

gold_rows = load_jsonl(
    ROOT / "00_FROZEN/benchmark_gold_v1.jsonl"
)

assert len(gold_rows) == 13

gold = {
    r["benchmark_id"]: r
    for r in gold_rows
}

# Give questions neutral aliases Q01 ... Q13
question_alias = {
    qid: f"Q{i:02d}"
    for i, qid in enumerate(sorted(gold), 1)
}

# --------------------------------------------------
# Gold reference shown to reviewer
# --------------------------------------------------

def make_gold_reference(g):
    parts = []

    for i, b in enumerate(g["branches"], 1):
        parts.append(
            f"[G{i}] {b.get('label','')}\n"
            f"Summary: {b.get('semantic_summary','')}\n"
            f"Claim: {b.get('source_claim','')}"
        )

        evs = b.get("evidence", [])

        for j, e in enumerate(evs, 1):
            text = e.get("text", "")
            paper = e.get("paper_title", "")
            parts.append(
                f"  Evidence {j}: {text}"
                + (f" | paper={paper}" if paper else "")
            )

    return "\n".join(parts)

# --------------------------------------------------
# Build a neutral source map
# --------------------------------------------------

SOURCE_RE = re.compile(r"^\[(T\d+|S\d+)\]\s*(.*)$")

def neutralize_sources(context):
    entries = []

    for line in context.splitlines():
        line = line.strip()

        # Do not expose Multi-branch grouping
        if re.match(r"^\[BRANCH\s+\d+\]", line):
            continue

        m = SOURCE_RE.match(line)

        if not m:
            continue

        old_id = m.group(1)
        payload = m.group(2).strip()

        # Remove KG type labels to reduce condition leakage
        payload = re.sub(
            r"^(SUMMARY|CLAIM|EVIDENCE):\s*",
            "",
            payload,
            flags=re.I
        )

        entries.append((old_id, payload))

    # Preserve source order but assign neutral IDs
    mapping = {
        old: f"R{i:02d}"
        for i, (old, _) in enumerate(entries, 1)
    }

    source_map = "\n".join(
        f"[{mapping[old]}] {payload}"
        for old, payload in entries
    )

    return mapping, source_map

def replace_source_ids(text, mapping):
    # Mechanical relabeling only; no semantic rewriting
    for old in sorted(mapping, key=len, reverse=True):
        text = re.sub(
            rf"\b{re.escape(old)}\b",
            mapping[old],
            text
        )
    return text

# --------------------------------------------------
# Load all 52 frozen answers and inputs
# --------------------------------------------------

blind_rows = []
secret_rows = []

for condition, stem in CONDITIONS.items():

    input_rows = {
        r["benchmark_id"]: r
        for r in load_jsonl(
            ROOT / "01_INPUTS" / f"{stem}.jsonl"
        )
    }

    output_rows = {
        r["benchmark_id"]: r
        for r in load_jsonl(
            ROOT / "04_OUTPUTS" / f"{stem}.jsonl"
        )
    }

    assert set(input_rows) == set(gold)
    assert set(output_rows) == set(gold)

    for qid in sorted(gold):

        inp = input_rows[qid]
        out = output_rows[qid]
        g = gold[qid]

        mapping, source_map = neutralize_sources(
            inp.get("context", "")
        )

        raw_answer = out["raw_output"]

        # Only mechanically rename source IDs.
        masked_answer = replace_source_ids(
            raw_answer,
            mapping
        )

        # Detect accidental explicit condition-name leakage.
        leakage_terms = [
            "LLM_ONLY",
            "TEXT_RAG",
            "FLAT_KG",
            "MULTIBRANCH_KG",
            "LLM-only",
            "Text-RAG",
            "Flat-KG",
            "Multi-branch KG",
        ]

        leakage_hits = [
            x for x in leakage_terms
            if x.lower() in masked_answer.lower()
        ]

        blind_rows.append({
            "_benchmark_id": qid,
            "_condition": condition,
            "_raw_sha256": hashlib.sha256(
                raw_answer.encode("utf-8")
            ).hexdigest(),

            "question_id": question_alias[qid],
            "question": g["question"],
            "deposit_name": g.get("deposit_name", ""),
            "dimension": g.get("dimension", ""),
            "gold_branch_count": g["semantic_branch_count"],
            "gold_branch_relation": g["gold_branch_relation"],
            "gold_reference": make_gold_reference(g),
            "source_map": (
                source_map
                if source_map
                else "(No external source context was provided.)"
            ),
            "answer": masked_answer,

            # Reviewer fills below
            "review_status": "",
            "correct_gold_branches_n": "",
            "branch_coverage": "",
            "single_answer_collapse": "",
            "relation_awareness": "",
            "substantive_claims_n": "",
            "unsupported_claims_n": "",
            "unsupported_claim_rate": "",
            "attributed_evidence_n": "",
            "correct_attributions_n": "",
            "evidence_attribution_accuracy": "",
            "uncertainty_quality": "",
            "reviewer_notes": "",

            "_leakage_hits": ";".join(leakage_hits),
        })

# --------------------------------------------------
# Randomize 52 answers globally
# --------------------------------------------------

rng = random.Random(SEED)
rng.shuffle(blind_rows)

for i, r in enumerate(blind_rows, 1):
    r["blind_answer_id"] = f"A{i:03d}"

# --------------------------------------------------
# Separate secret key BEFORE writing review file
# --------------------------------------------------

for r in blind_rows:
    secret_rows.append({
        "blind_answer_id": r["blind_answer_id"],
        "question_id": r["question_id"],
        "benchmark_id": r["_benchmark_id"],
        "condition": r["_condition"],
        "raw_output_sha256": r["_raw_sha256"],
        "possible_explicit_condition_leakage":
            r["_leakage_hits"],
    })

# Public reviewer columns
review_fields = [
    "blind_answer_id",
    "question_id",
    "question",
    "deposit_name",
    "dimension",
    "gold_branch_count",
    "gold_branch_relation",
    "gold_reference",
    "source_map",
    "answer",

    "review_status",
    "correct_gold_branches_n",
    "branch_coverage",
    "single_answer_collapse",
    "relation_awareness",
    "substantive_claims_n",
    "unsupported_claims_n",
    "unsupported_claim_rate",
    "attributed_evidence_n",
    "correct_attributions_n",
    "evidence_attribution_accuracy",
    "uncertainty_quality",
    "reviewer_notes",
]

review_path = OUT / "blind_review_52_FROZEN.csv"

with review_path.open(
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:
    w = csv.DictWriter(
        f,
        fieldnames=review_fields
    )
    w.writeheader()

    for r in blind_rows:
        w.writerow({
            k: r[k]
            for k in review_fields
        })

# Make editable working copy.
working_path = OUT / "blind_review_52_WORKING.csv"
shutil.copyfile(review_path, working_path)

# Secret key
key_path = OUT / "blind_condition_key_SECRET.csv"

with key_path.open(
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:
    w = csv.DictWriter(
        f,
        fieldnames=secret_rows[0].keys()
    )
    w.writeheader()
    w.writerows(secret_rows)

# --------------------------------------------------
# COC four-question list
# --------------------------------------------------

coc = [
    g for g in gold_rows
    if g["gold_branch_relation"]
       == "COMPLEMENTARY_OR_CONVERGENT"
]

coc_path = OUT / "COMPLEMENTARY_OR_CONVERGENT_questions.csv"

with coc_path.open(
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:
    fields = [
        "question_id",
        "benchmark_id",
        "question",
        "gold_branch_relation"
    ]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()

    for g in coc:
        w.writerow({
            "question_id":
                question_alias[g["benchmark_id"]],
            "benchmark_id":
                g["benchmark_id"],
            "question":
                g["question"],
            "gold_branch_relation":
                g["gold_branch_relation"],
        })

# --------------------------------------------------
# Reviewer instructions
# --------------------------------------------------

rubric_path = OUT / "BLIND_REVIEW_INSTRUCTIONS.md"

rubric_path.write_text(
"""# Condition-masked review instructions

IMPORTANT:
- Score only blind_review_52_WORKING.csv.
- Do NOT open blind_condition_key_SECRET.csv until all 52 answers are scored.
- Do NOT change the answer text.
- Gold reference is visible; experimental condition is hidden.

## Branch Coverage
Fill:
correct_gold_branches_n

Then:
branch_coverage =
correct_gold_branches_n / gold_branch_count

A branch counts as recovered when the answer expresses its core geological
interpretation correctly. Exact wording is not required.

## Single-answer Collapse
1 = answer represents no more than one Gold semantic branch when Gold has
multiple branches.
0 = otherwise.

## Relation Awareness
1.0 = correct relation structure
0.5 = partially correct
0.0 = absent or incorrect

For COMPLEMENTARY_OR_CONVERGENT:
1.0 = recognizes branches as non-competing and correctly explains either
      complementary process/aspect constraints or convergent evidence.
0.5 = recognizes compatibility or gives one admissible relation label but
      explanation is incomplete.
0.0 = treats branches as competing/contradictory/mutually exclusive, or
      fails to recognize their relationship.

## Unsupported Claim Rate
Identify substantive geological interpretations in the answer.

unsupported_claim_rate =
unsupported_claims_n / substantive_claims_n

Do not count harmless wording differences as unsupported claims.

## Evidence Attribution Accuracy
For answers that attribute/cite provided sources:

evidence_attribution_accuracy =
correct_attributions_n / attributed_evidence_n

Use the neutral Rxx source_map shown in each row.

If attributed_evidence_n = 0:
leave evidence_attribution_accuracy blank rather than forcing 0.

## Uncertainty Quality
2 = appropriate qualification of evidence strength/boundaries
1 = partial qualification
0 = absent or misleading qualification

## Review status
Use:
DONE
or
NEEDS_CONTEXT

Do not infer or record the experimental condition during scoring.
""",
    encoding="utf-8"
)

# --------------------------------------------------
# Package manifest + frozen hashes
# --------------------------------------------------

explicit_leak_rows = [
    x for x in secret_rows
    if x["possible_explicit_condition_leakage"]
]

manifest = {
    "status": "CONDITION_MASKED_PACKAGE_FROZEN",
    "random_seed": SEED,
    "n_questions": 13,
    "n_conditions": 4,
    "n_answers": 52,
    "source_id_masking":
        "Txx/Sxx mechanically relabeled to neutral Rxx IDs.",
    "context_masking":
        "BRANCH headers and KG atom-type prefixes removed from reviewer source map.",
    "semantic_answer_rewriting":
        False,
    "explicit_condition_leakage_rows":
        len(explicit_leak_rows),
    "coc_question_count":
        len(coc),
}

manifest_path = OUT / "BLIND_PACKAGE_MANIFEST.json"
manifest_path.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)

hash_targets = [
    review_path,
    key_path,
    coc_path,
    rubric_path,
    manifest_path,
]

hash_path = OUT / "BLIND_PACKAGE_SHA256SUMS.txt"

with hash_path.open("w", encoding="utf-8") as f:
    for p in hash_targets:
        f.write(
            f"{sha256_file(p)}  {p.name}\n"
        )

print("=" * 72)
print("BLIND PACKAGE CREATED")
print("=" * 72)
print("Answers:", len(blind_rows))
print("Questions:", len(gold))
print("COC questions:", len(coc))
print(
    "Explicit condition leakage rows:",
    len(explicit_leak_rows)
)

print("\nIMPORTANT:")
print("Score ONLY:")
print(" ", working_path)
print("\nDO NOT OPEN UNTIL SCORING COMPLETE:")
print(" ", key_path)

print("\nCreated:")
for p in [
    review_path,
    working_path,
    key_path,
    coc_path,
    rubric_path,
    manifest_path,
    hash_path,
]:
    print(" ", p)

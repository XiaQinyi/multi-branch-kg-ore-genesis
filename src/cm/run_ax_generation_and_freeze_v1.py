#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/root/autodl-tmp/Geology_CE_ModelReady_v2")
ap.add_argument("--model", default=None, help="Local Qwen2.5-7B(-Instruct) path or Hugging Face model id. Can also set QWEN_MODEL.")
a = ap.parse_args()

ROOT = Path(a.root).resolve()
EXP = ROOT / "MultiBranch_LLM_Experiment_v1"
BASE = ROOT / "runs/multibranch_kg_v2_grounded/20_AX_Prospective_Replication_v1"
INP = BASE / "00_FROZEN_INPUTS"
OUT = BASE / "01_FROZEN_OUTPUTS"
WORK = BASE / "_WORK_GENERATION_V1"
RUNNER = EXP / "03_SCRIPTS/run_hf.py"

SEED = 20260808
TEMP = 0.0
MAX_NEW = 900
EXPECTED_IDS = [
    "AX01_KEYIN_PEGMATITE_GENESIS",
    "AX02_KEYIN_COUNTRYROCK_LI_ENRICHMENT",
    "AX03_DAHONGLIUTAN_PEGMATITE_GENESIS",
    "AX04_SHIZILING_GRANITE_TYPE",
    "AX05_LIF_GRANITE_MINERAL_GENESIS",
]

def stop(msg):
    raise SystemExit("HARD STOP — " + msg)

def sha256(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

for p in [
    INP / "90_FROZEN_MANIFEST.json",
    INP / "SHA256SUMS.txt",
    INP / "flat_kg.jsonl",
    INP / "multibranch_kg.jsonl",
    RUNNER,
]:
    if not p.exists():
        stop(f"missing required frozen input/source: {p}")

if OUT.exists():
    stop(f"refuse to overwrite frozen outputs: {OUT}")
if WORK.exists():
    stop(f"work directory already exists from an earlier attempt: {WORK}")

manifest_in = json.loads((INP / "90_FROZEN_MANIFEST.json").read_text(encoding="utf-8"))
if manifest_in.get("status") != "FROZEN_AX_INPUTS_V1":
    stop("input manifest status is not FROZEN_AX_INPUTS_V1")

# Verify frozen input hashes before generation.
for line in (INP / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    h, name = line.split("  ", 1)
    p = INP / name
    if not p.exists() or sha256(p) != h:
        stop(f"frozen input hash verification failed: {name}")

def read_jsonl(p):
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

for name, cond in [("flat_kg.jsonl", "FLAT_KG"), ("multibranch_kg.jsonl", "MULTIBRANCH_KG")]:
    rows = read_jsonl(INP / name)
    if len(rows) != 5:
        stop(f"{name} must contain 5 rows, got {len(rows)}")
    ids = [r.get("benchmark_id") for r in rows]
    if ids != EXPECTED_IDS:
        stop(f"{name} benchmark order/IDs differ: {ids}")
    if any(r.get("condition") != cond for r in rows):
        stop(f"{name} contains wrong condition")

# Resolve model.
model = a.model or os.environ.get("QWEN_MODEL")
if not model:
    snap_root = Path("/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots")
    snaps = sorted([p for p in snap_root.glob("*") if (p / "config.json").exists()]) if snap_root.exists() else []
    if len(snaps) == 1:
        model = str(snaps[0])
    elif len(snaps) > 1:
        model = str(snaps[-1])

if not model:
    stop(
        "Qwen model path not resolved. Re-run with --model /path/to/Qwen2.5-7B-Instruct "
        "or set QWEN_MODEL. No generation has been performed."
    )

# If a local-looking path was supplied, require it to exist.
if model.startswith("/") and not Path(model).exists():
    stop(f"model path does not exist: {model}")

WORK.mkdir(parents=True)
flat_work = WORK / "flat_kg.jsonl"
multi_work = WORK / "multibranch_kg.jsonl"

cmd_common = [
    sys.executable, str(RUNNER),
    "--model", model,
    "--max-new-tokens", str(MAX_NEW),
    "--temperature", str(TEMP),
    "--seed", str(SEED),
]

print("=" * 100)
print("AX PROSPECTIVE GENERATION")
print("=" * 100)
print("MODEL =", model)
print("RUNNER =", RUNNER)
print("RUNNER_SHA256 =", sha256(RUNNER))
print("SEED =", SEED)
print("TEMPERATURE =", TEMP)
print("MAX_NEW_TOKENS =", MAX_NEW)
print("EXPECTED_OUTPUTS = 10")
print("IMPORTANT: raw generated outputs will be frozen before evaluation.")

try:
    subprocess.run(
        cmd_common + [
            "--input", str(INP / "flat_kg.jsonl"),
            "--output", str(flat_work),
        ],
        check=True,
    )
    subprocess.run(
        cmd_common + [
            "--input", str(INP / "multibranch_kg.jsonl"),
            "--output", str(multi_work),
        ],
        check=True,
    )
except Exception:
    print("\nGENERATION DID NOT COMPLETE.")
    print("Partial work, if any, remains at:", WORK)
    print("No frozen output directory was created.")
    raise

flat_rows = read_jsonl(flat_work)
multi_rows = read_jsonl(multi_work)

def completeness(rows, cond):
    if len(rows) != 5:
        stop(f"{cond}: expected 5 output rows, got {len(rows)}; raw work retained at {WORK}")
    ids = [r.get("benchmark_id") for r in rows]
    if ids != EXPECTED_IDS:
        stop(f"{cond}: output benchmark IDs/order mismatch: {ids}")
    if any(r.get("condition") != cond for r in rows):
        stop(f"{cond}: wrong condition label in generated outputs")
    if any(not str(r.get("raw_output", "")).strip() for r in rows):
        stop(f"{cond}: empty raw_output detected")

completeness(flat_rows, "FLAT_KG")
completeness(multi_rows, "MULTIBRANCH_KG")

# Structural validation is descriptive only. We freeze the original generated text
# even if one answer is malformed JSON; we do NOT regenerate for cosmetic validity.
validation = []
required_top = {"answer", "interpretations", "overall_relation", "uncertainty"}
required_interp = {"name", "summary", "evidence_refs", "relation_to_others"}

for r in flat_rows + multi_rows:
    raw = str(r["raw_output"])
    ok_json = False
    ok_schema = False
    err = ""
    try:
        obj = json.loads(raw)
        ok_json = True
        ok_schema = (
            isinstance(obj, dict)
            and required_top.issubset(obj.keys())
            and isinstance(obj.get("interpretations"), list)
            and all(isinstance(x, dict) and required_interp.issubset(x.keys()) for x in obj.get("interpretations", []))
        )
        if not ok_schema:
            err = "JSON parsed but expected output fields are incomplete"
    except Exception as ex:
        err = f"{type(ex).__name__}: {ex}"
    validation.append({
        "benchmark_id": r["benchmark_id"],
        "condition": r["condition"],
        "raw_output_chars": len(raw),
        "json_parse_ok": ok_json,
        "schema_fields_ok": ok_schema,
        "validation_note": err,
    })

# Freeze exactly the generated rows after 10/10 completeness.
OUT.mkdir(parents=True)
p_flat = OUT / "10_FLAT_KG_OUTPUTS_5.jsonl"
p_multi = OUT / "11_MULTIBRANCH_KG_OUTPUTS_5.jsonl"
p_combined = OUT / "12_OUTPUTS_10_COMBINED.jsonl"
shutil.copy2(flat_work, p_flat)
shutil.copy2(multi_work, p_multi)

with p_combined.open("w", encoding="utf-8") as f:
    for r in flat_rows + multi_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

p_val = OUT / "20_OUTPUT_STRUCTURE_VALIDATION.tsv"
with p_val.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "benchmark_id", "condition", "raw_output_chars",
            "json_parse_ok", "schema_fields_ok", "validation_note"
        ],
        delimiter="\t",
        lineterminator="\n",
    )
    w.writeheader()
    w.writerows(validation)

out_manifest = {
    "status": "FROZEN_AX_OUTPUTS_V1",
    "freeze_date": "2026-08-20",
    "prospective_replication": True,
    "benchmark_items": 5,
    "conditions": ["FLAT_KG", "MULTIBRANCH_KG"],
    "output_rows": 10,
    "model_argument": model,
    "generation": {
        "temperature": TEMP,
        "seed": SEED,
        "max_new_tokens": MAX_NEW,
        "runner_path": str(RUNNER.relative_to(ROOT)),
        "runner_sha256": sha256(RUNNER),
        "flat_input_sha256": sha256(INP / "flat_kg.jsonl"),
        "multibranch_input_sha256": sha256(INP / "multibranch_kg.jsonl"),
        "input_manifest_sha256": sha256(INP / "90_FROZEN_MANIFEST.json"),
    },
    "freeze_rule": "Raw outputs frozen immediately after 10/10 generation completeness and before any semantic scoring or unblinding.",
    "structural_validation_policy": "Malformed JSON, if present, is recorded but is not regenerated or cosmetically repaired.",
    "reporting_guardrails": [
        "Report independently from Experiment I 52 outputs.",
        "Do not merge AX01–AX05 with historical A01–A03 as n=8.",
        "Treat as prospective replication, not a strong confirmatory significance test.",
    ],
}
p_manifest = OUT / "90_FROZEN_MANIFEST.json"
p_manifest.write_text(json.dumps(out_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt")
p_sha = OUT / "SHA256SUMS.txt"
with p_sha.open("w", encoding="utf-8") as f:
    for p in files:
        f.write(f"{sha256(p)}  {p.name}\n")

for line in p_sha.read_text(encoding="utf-8").splitlines():
    h, name = line.split("  ", 1)
    if sha256(OUT / name) != h:
        stop(f"frozen output SHA verification failed: {name}")

# Remove temporary generation work only after byte copies and hashes exist.
shutil.rmtree(WORK)

print("\n" + "=" * 100)
print("AX OUTPUT FREEZE COMPLETE")
print("=" * 100)
print("OUT =", OUT)
print("FLAT_ROWS = 5")
print("MULTI_ROWS = 5")
print("TOTAL_ROWS = 10")
print("JSON_PARSE_OK =", sum(1 for x in validation if x["json_parse_ok"]), "/ 10")
print("SCHEMA_FIELDS_OK =", sum(1 for x in validation if x["schema_fields_ok"]), "/ 10")
print("\nSHA256SUMS:")
print(p_sha.read_text(encoding="utf-8"), end="")
print("FINAL_STATUS = FROZEN_AX_OUTPUTS_V1")
print("READY_FOR_BLIND_EVALUATION = TRUE")

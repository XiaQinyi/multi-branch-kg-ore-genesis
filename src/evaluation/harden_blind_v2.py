#!/usr/bin/env python3

import csv
import hashlib
import random
import re
import shutil
from pathlib import Path

ROOT = Path("Paper_Final_Evaluation_v1/02_Blind_Eval")
SRC = ROOT / "blind_review_52_FROZEN.csv"

OUT_FROZEN = ROOT / "blind_review_52_FROZEN_v2.csv"
OUT_WORKING = ROOT / "blind_review_52_WORKING_v2.csv"
OUT_HASH = ROOT / "BLIND_V2_SHA256.txt"

SEED = 20260808

with SRC.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
    fields = list(rows[0].keys())

def replace_refs(text, mapping):
    # temporary placeholders prevent R01 -> R02 cascading
    tmp = {}
    for i, old in enumerate(mapping):
        key = f"__REF_TMP_{i:03d}__"
        tmp[old] = key
        text = re.sub(rf"\b{re.escape(old)}\b", key, text)

    for old, key in tmp.items():
        text = text.replace(key, mapping[old])

    return text

for row in rows:
    source_map = row["source_map"].strip()

    if source_map == "(No external source context was provided.)":
        continue

    entries = []

    for line in source_map.splitlines():
        m = re.match(r"^\[(R\d+)\]\s*(.*)$", line.strip())
        if m:
            entries.append((m.group(1), m.group(2)))

    # deterministic per-answer shuffle
    seed_text = f"{SEED}|{row['blind_answer_id']}"
    local_seed = int(
        hashlib.sha256(seed_text.encode()).hexdigest()[:16],
        16
    )

    rng = random.Random(local_seed)
    rng.shuffle(entries)

    mapping = {
        old: f"R{i:02d}"
        for i, (old, _) in enumerate(entries, 1)
    }

    row["source_map"] = "\n".join(
        f"[{mapping[old]}] {payload}"
        for old, payload in entries
    )

    row["answer"] = replace_refs(
        row["answer"],
        mapping
    )

with OUT_FROZEN.open(
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

shutil.copyfile(
    OUT_FROZEN,
    OUT_WORKING
)

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

OUT_HASH.write_text(
    f"{sha256(OUT_FROZEN)}  {OUT_FROZEN.name}\n",
    encoding="utf-8"
)

print("V2 CREATED")
print("rows =", len(rows))
print("frozen =", OUT_FROZEN)
print("working =", OUT_WORKING)
print("sha256 =", sha256(OUT_FROZEN))

# Multi-Branch Knowledge Graph (Multi-KG): Supplementary Code for Peer Review

**Accompanying manuscript:** *A Multi-Branch Knowledge Graph for LLM-Enhanced Reasoning: Application to Ore-Genesis Questions with Multiple Interpretations*

## 1. Overview and Scope

This repository provides supplementary source code and a self-contained synthetic example to support editorial and peer review of the accompanying manuscript. It enables reviewers to:

- Inspect graph-construction, LLM-inference, and evaluation scripts related to the study.
- Verify the graph schema, structural invariants, and Flat-KG vs. Multi-KG prompt-parity pipeline locally on a standard CPU without external dependencies or restricted data.

**Scope note:** To respect third-party publication copyrights and data-access restrictions, this repository does not include the full curated research knowledge graph, source literature corpus, restricted annotations, model weights, or frozen experimental outputs. Consequently, the bundled synthetic example is intended for code and schema verification rather than end-to-end reproduction of the manuscript's quantitative results. Full details are documented in [`DATA_AVAILABILITY_STATEMENT.txt`](DATA_AVAILABILITY_STATEMENT.txt).

## 2. Quick Start (CPU-Only, Standard Library)

From the **root directory of this GitHub repository**, run the verification workflow using **Python 3.10 or newer** (standard library only; no GPU or model downloads required):

```bash
python3 examples/synthetic_graph/verify_example.py --expected-nodes 28 --expected-edges 35
python3 examples/synthetic_graph/make_llm_inputs.py --output-dir ./example_outputs/graph_inputs
python3 demo/smoke_test.py
```

**Expected outcomes:**

- `verify_example.py` reports `PASSED` with 28 nodes, 35 edges, and zero dangling relations.
- `make_llm_inputs.py` processes three synthetic cases, reports exact atom parity between Flat-KG and Multi-KG inputs for each case, and writes `flat_kg.jsonl`, `multibranch_kg.jsonl`, and `parity_audit.json` to `./example_outputs/graph_inputs/`.
- `demo/smoke_test.py` reports `PASS` after building and validating an additional small synthetic graph, generating paired prompt records, and validating a synthetic model-output record.

To inspect the parity audit output:

```bash
python3 -m json.tool ./example_outputs/graph_inputs/parity_audit.json
```

An independent demonstration of Flat-KG and Multi-KG input formatting can also be run via:

```bash
python3 demo/build_synthetic_llm_inputs.py --output-dir ./example_outputs/format_example
```

These checks exercise the data pipeline without invoking Qwen inference. The results of checks on the bundled example are recorded in `EXAMPLE_VALIDATION_REPORT.json`.

## 3. Synthetic Example Graph

The `examples/synthetic_graph/` directory provides a miniature fictional dataset following the study's 12-column graph schema:

| File | Description |
| --- | --- |
| `nodes.csv` | 28 synthetic nodes in the 12-column node format. |
| `edges.csv` | 35 synthetic directed relations in the 12-column edge format. |
| `cases.jsonl` | Three synthetic ore-genesis-style question records. |
| `verify_example.py` | Validates schema integrity, unique node IDs, relation endpoints/directions, and expected counts. |
| `make_llm_inputs.py` | Converts identical claim/evidence atoms into Flat-KG and branch-grouped Multi-KG prompt records. |

The synthetic graph includes three `HYPOTHESIS_FAMILY` nodes and six `HYPOTHESIS_BRANCH` nodes, alongside `PAPER`, `DEPOSIT`, `CLAIM`, and `EVIDENCE` nodes connected via `STUDIES`, `ASSERTS`, `REPORTS`, `HAS_BRANCH`, `CENTRAL_CLAIM`, `SUPPORTS`, `OPPOSES`, and `QUALIFIES`.

**Schema semantics:** In this example, `OPPOSES` encodes an **evidence-to-claim** relation rather than an expert-confirmed branch-to-branch competition label; branch-level competition status remains `UNRESOLVED`. The generated Flat-KG and Multi-KG prompts contain identical claim/evidence atoms and differ in their structural grouping. This lightweight example illustrates the representation and formatting logic rather than the full retrieval or scientific adjudication workflow.

## 4. Repository Structure

| Path | Description and Execution Notes |
| --- | --- |
| `src/legacy_kg/build_multibranch_kg.py` | Constructs the claim/evidence graph using exact-text candidate grouping (does not infer expert-confirmed competing interpretations). |
| `src/legacy_kg/validate_kg_v1.py` | Checks structural and relational invariants of the graph schema. |
| `src/legacy_kg/prepare_existing_structured_candidates.py` | Prepares structured candidate records; execution on full data requires restricted study inputs. |
| `src/legacy_kg/infer_3seed_ensemble.py` | Multi-seed relation prediction script; requires trained checkpoints and study inputs not bundled here. |
| `src/llm/run_hf.py`, `src/llm/run_all_hf.py` | Hugging Face inference scripts for single-condition or four-condition runs; full runs require local model weights and original frozen inputs. |
| `src/evaluation/` | Utilities for output JSON validation, structural scoring, token accounting, and blinded evaluation-package assembly (separate from human expert accuracy assessment). |
| `src/cm/` | Multi-interpretation input and paired-analysis scripts provided for methodological inspection. |
| `demo/` | Standalone synthetic graph construction, validation, and prompt-formatting demos. |
| `configs/` | Reference configuration files and dependency specifications from the study workflow. |
| `PROVENANCE.csv` | Maps archived scripts to original source locations, checksums, and peer-review sanitization changes. |
| `EXAMPLE_VALIDATION_REPORT.json` | Validation record for the self-contained synthetic example. |
| `DATA_AVAILABILITY_STATEMENT.txt` | Summary of material scope and access conditions. |

**Notes on archived study scripts:**

- **Confidential paper ID redaction (`src/legacy_kg/`):** Confidential held-out paper identifiers have been removed from the shared copies. Running these scripts on real, non-demo study datasets requires an authorized `MULTIKG_LOCKED_PAPER_IDS` environment variable; scripts fail closed when that variable is required but unset. Checksums and edit records are logged in `PROVENANCE.csv`.
- **Methodological reference scripts (`src/cm/`):** These scripts retain internal `AX` case identifiers (not cross-mapped to manuscript `CM` IDs in this repository) and environment-specific paths from the original compute server. They are included for code inspection rather than direct local execution.

## 5. Optional Local LLM Inference

The Quick Start workflow requires no third-party packages (`requirements.txt`). For reviewers who wish to test the Hugging Face inference interface locally on synthetic prompts, `requirements-llm.txt` lists indicative (unpinned) dependencies. Running inference requires a compatible local model and appropriate hardware:

```bash
python3 -m pip install -r requirements-llm.txt
python3 src/llm/run_hf.py \
  --model /path/to/compatible/Qwen2.5-7B-Instruct \
  --input ./example_outputs/graph_inputs/multibranch_kg.jsonl \
  --output ./example_outputs/llm_predictions.jsonl \
  --max-new-tokens 900 --temperature 0 --seed 20260808
```

This command illustrates the inference interface on synthetic inputs rather than replicating the manuscript's full experimental runs. The historical script may load third-party model code; reviewers should use a trusted checkpoint.

## 6. Data Availability and Limitations

The following study artifacts are not distributed in this repository due to third-party publication copyrights and data-access restrictions: the full curated knowledge graph, scientific text extracted from published papers, restricted claim/evidence annotations, original H/CM question sets, expert evaluation labels, frozen model outputs, and model weights. Accordingly, this package supports inspection of the methodological pipeline and schema behavior, rather than independent verification of the manuscript's aggregate graph statistics or empirical performance metrics.

Please consult [`DATA_AVAILABILITY_STATEMENT.txt`](DATA_AVAILABILITY_STATEMENT.txt) for details. Inquiries regarding restricted materials or verification of numerical results may be directed to the corresponding author via the journal's editorial office.

## 7. Code Availability and Licensing

This repository is publicly accessible to support editorial review and methodological inspection. An open-source license has not yet been confirmed by the relevant code rightsholders. Public visibility does not by itself grant permission to copy, modify, or redistribute the software. Licensing information will be updated following the necessary author and institutional approvals. Restricted research data are not released under any software license associated with this repository.

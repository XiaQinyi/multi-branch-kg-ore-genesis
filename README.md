# Multi-Branch Knowledge Graph (Multi-KG): Supplementary Code for Peer Review

**Accompanying manuscript:** *A Multi-Branch Knowledge Graph for LLM-Enhanced Reasoning: Application to Ore-Genesis Questions with Multiple Interpretations*

## 1. Purpose and scope

This archive is provided as **supplementary source code and a runnable synthetic example for editorial and peer review**. It contains selected historical scripts for knowledge-graph construction, model inference and evaluation, together with a deliberately small fictional graph that enables reviewers to inspect the expected data structure and run independent checks without a GPU or private research materials.

**Important scope distinction:** This is **not** the full original research knowledge graph, the complete training/evaluation corpus, or an end-to-end reproduction of the manuscript's quantitative results. Some included scripts document historical stages and require study-specific private inputs, checkpoints or configuration that are not supplied. The sample graph is invented for testing; it must not be presented as an actual geological dataset or independent evidence supporting the paper's conclusions.

## 2. Quick start: tested, no GPU or downloads required

Open a terminal in the extracted `MultiKG_Peer_Review_Materials` directory. Install **Python 3.10 or newer**; the following commands use only the Python standard library:

```bash
python3 examples/synthetic_graph/verify_example.py --expected-nodes 28 --expected-edges 35
python3 examples/synthetic_graph/make_llm_inputs.py --output-dir ./example_outputs/graph_inputs
python3 demo/smoke_test.py
```

Expected results:

- The graph validator reports `PASSED`, 28 nodes, 35 edges, and zero dangling relations.
- The input builder reports three example cases and `flat_multi_atom_parity: EXACT` for each, and creates `flat_kg.jsonl`, `multibranch_kg.jsonl`, and `parity_audit.json` in `example_outputs/graph_inputs/`.
- The CPU smoke test reports `PASS` for building and validating another fictional small graph, preparing prompt records, and checking a synthetic model-output record.

To inspect the paired input records after generation:

```bash
python3 -m json.tool ./example_outputs/graph_inputs/parity_audit.json
```

There is also an independent demonstration of Flat-KG and Multi-KG input formatting:

```bash
python3 demo/build_synthetic_llm_inputs.py --output-dir ./example_outputs/format_example
```

These commands do **not** execute actual Qwen inference or reproduce manuscript performance estimates. See `EXAMPLE_VALIDATION_REPORT.json` for a test log prepared with this archive.

## 3. The provided example graph

The `examples/synthetic_graph/` folder contains:

| File | Description |
| --- | --- |
| `nodes.csv` | 28 fictional nodes using the 12-column legacy graph node format. |
| `edges.csv` | 35 fictional directed relations using the 12-column legacy edge format. |
| `cases.jsonl` | Three fictional ore-genesis-style question records. |
| `verify_example.py` | Checks schema, unique node IDs, relation endpoints/directions and expected counts. |
| `make_llm_inputs.py` | Converts identical claim/evidence atoms to Flat-KG and branch-grouped Multi-KG prompt records. |

The example contains three `HYPOTHESIS_FAMILY` nodes and six `HYPOTHESIS_BRANCH` nodes. Other node types include `PAPER`, `DEPOSIT`, `CLAIM` and `EVIDENCE`. Relations include `STUDIES`, `ASSERTS`, `REPORTS`, `HAS_BRANCH`, `CENTRAL_CLAIM`, `SUPPORTS`, `OPPOSES` and `QUALIFIES`.

In the example, `OPPOSES` represents an **evidence-to-claim** relationship. It is **not** a manually confirmed branch-to-branch competition label; branch competition remains `UNRESOLVED`. The Flat-KG and Multi-KG prompts contain the same claim/evidence content, with different grouping/presentation only. This miniature example does not reimplement the full retrieval or scientific adjudication workflow.

## 4. Guide to source files

| Location | Function / constraints |
| --- | --- |
| `src/legacy_kg/build_multibranch_kg.py` | Builds a legacy claim/evidence graph with exact-text candidate grouping; does not infer expert-confirmed competing interpretations. |
| `src/legacy_kg/validate_kg_v1.py` | Checks structural and relation invariants of the legacy graph. |
| `src/legacy_kg/prepare_existing_structured_candidates.py` | Prepares structured candidate records; real runs require restricted study inputs. |
| `src/legacy_kg/infer_3seed_ensemble.py` | Historical multi-seed relation prediction; requires checkpoints and non-included inputs. |
| `src/llm/run_hf.py`, `src/llm/run_all_hf.py` | Hugging Face inference scripts for one or four experimental conditions; original frozen inputs and model weights are not bundled. |
| `src/evaluation/` | Output JSON checks, structural scoring, token accounting and preparation of blinded evaluation packages. These do not independently establish expert judgment accuracy. |
| `src/cm/` | Historical AX-related multi-interpretation input and paired-analysis scripts. Their AX IDs have not been independently mapped to the paper's CM IDs. |
| `demo/` | Runnable synthetic build/validation and input-formatting demonstrations. |
| `configs/` | Historical configuration/requirements examples. |
| `PROVENANCE.csv` | Maps selected source scripts to historical locations, checksums and documented reviewer-copy modifications. |
| `EXAMPLE_VALIDATION_REPORT.json` | Record of validation performed on this self-contained example. |
| `DATA_AVAILABILITY_STATEMENT.txt` | Boundaries and restrictions of the materials supplied for peer review. |

The `src/legacy_kg/` copies include a safety change: confidential held-out paper identifiers are not embedded. Running those scripts with **real, non-demo study data** requires a separately provided authorized `MULTIKG_LOCKED_PAPER_IDS` environment setting. The scripts fail closed when that setting is required but absent. The actual identifiers are not in this archive. See `PROVENANCE.csv` for the historical source and edited-copy hashes.

Some historical `src/cm/` scripts still assume server-specific paths. They are supplied for methodological inspection, **not** guaranteed runnable on a reviewer's computer with no original data. Do not run them against study directories or existing frozen outputs; they may create or overwrite artifacts.

## 5. Optional model inference (not needed for the smoke test)

`requirements.txt` has no mandatory third-party dependencies for the synthetic workflow. `requirements-llm.txt` lists indicative, **not fully pinned or independently environment-verified**, packages for optional Hugging Face inference. A compatible model and local compute resources are needed. For illustration only, after running the Quick start:

```bash
python3 -m pip install -r requirements-llm.txt
python3 src/llm/run_hf.py \
  --model /path/to/compatible/Qwen2.5-7B-Instruct \
  --input ./example_outputs/graph_inputs/multibranch_kg.jsonl \
  --output ./example_outputs/llm_predictions.jsonl \
  --max-new-tokens 900 --temperature 0 --seed 20260808
```

This optional command has **not** been validated as a faithful reconstruction of the manuscript's generation experiments.

## 6. Data and interpretation limitations

The full curated original graph, scientific text extracted from third-party publications, restricted claim/evidence annotations, original H/CM question sets, expert evaluation labels, frozen model outputs and model weights are **not included** in this peer-review ZIP. Consequently the example cannot reconstruct the full graph or verify the manuscript's reported graph statistics and performance metrics. Reviewers should distinguish the schema/functionality demonstration from the research evidence itself.

Consult `DATA_AVAILABILITY_STATEMENT.txt` for the precise supply scope. Any request for restricted original materials or for verification of numerical findings should be handled through the authors and journal's editorial process; availability is not promised by this archive.

## 7. Access and permissions

This copy is supplied for assessment in connection with the manuscript. **No public open-source license is asserted or included**, and the archive does not grant public redistribution or reuse rights. It should not be described as a public repository deposit. Any public source release, license and persistent archive are separate decisions requiring rightsholder and journal approval. The contents are not claimed to satisfy any particular journal's full code/data-deposition policy by themselves.

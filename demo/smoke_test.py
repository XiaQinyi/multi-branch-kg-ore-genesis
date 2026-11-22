#!/usr/bin/env python3
"""Safe, CPU-only illustration using wholly synthetic data; no model weights needed."""
from pathlib import Path
import csv, json, subprocess, sys, tempfile
ROOT=Path(__file__).resolve().parents[1]
DEMO=ROOT/'demo'
PY=sys.executable

def run(args):
    completed=subprocess.run([PY,*map(str,args)],cwd=ROOT,text=True,capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"FAILED {args}:\n{completed.stdout}\n{completed.stderr}")
    return completed.stdout

with tempfile.TemporaryDirectory(prefix='multikg-fictional-') as temp:
    td=Path(temp);kg=td/'kg';cand=td/'candidates';inf=td/'inference'
    cand.mkdir();inf.mkdir()
    spans=DEMO/'data/synthetic_spans.jsonl';pred=DEMO/'data/synthetic_predictions.jsonl'
    (cand/'corpus_spans.jsonl').write_bytes(spans.read_bytes())
    (inf/'ensemble_relation_predictions.jsonl').write_bytes(pred.read_bytes())
    run([ROOT/'src/legacy_kg/build_multibranch_kg.py','--predictions',pred,'--span_dimensions',spans,'--output_dir',kg])
    run([ROOT/'src/legacy_kg/validate_kg_v1.py','--candidate_dir',cand,'--inference_dir',inf,'--kg_dir',kg,'--output',td/'check.json'])
    check=json.loads((td/'check.json').read_text())
    assert check['status']=='PASSED',check
    nodes=list(csv.DictReader((kg/'nodes.csv').open(encoding='utf-8-sig')))
    edges=list(csv.DictReader((kg/'edges.csv').open(encoding='utf-8-sig')))
    assert len(nodes)>0 and len(edges)>0
    # The legacy KG sets competition_status=UNRESOLVED; the synthetic semantic relation is a separate illustration.
    branches=[x for x in nodes if x['type']=='HYPOTHESIS_BRANCH']
    assert len(branches)==2 and all(x['competition_status']=='UNRESOLVED' for x in branches)
    run([DEMO/'build_synthetic_llm_inputs.py','--output-dir',td/'llm_inputs'])
    assert (td/'llm_inputs/flat_kg.jsonl').is_file()
    assert (td/'llm_inputs/multibranch_kg.jsonl').is_file()
    valid=run([ROOT/'src/evaluation/validate_outputs.py','--input',DEMO/'data/synthetic_structural_output.jsonl'])
    assert 'VALID_JSON 1 / 1' in valid,valid
    structural=run([ROOT/'src/evaluation/score_structural.py','--input',DEMO/'data/synthetic_structural_output.jsonl'])
    assert 'DEMO-Q01' in structural,structural
    print('PASS: fictional KG builder/validator, input formatting, JSON structural check')
    print('Demo KG nodes:',len(nodes),'edges:',len(edges),'legacy branches:',len(branches))
    print('No private study data, real Qwen inference, or paper scores were used.')

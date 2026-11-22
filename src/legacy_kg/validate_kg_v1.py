#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import os
LOCKED = {item.strip() for item in os.environ.get("MULTIKG_LOCKED_PAPER_IDS", "").split(",") if item.strip()}
def _require_locked_ids(paper_ids):
    if not LOCKED and any(pid and not str(pid).startswith("DEMO-PAPER-") for pid in paper_ids):
        raise RuntimeError("Set MULTIKG_LOCKED_PAPER_IDS before processing non-demo paper IDs; never publish this private exclusion list")

def loadj(p): return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--candidate_dir',required=True); ap.add_argument('--inference_dir',required=True); ap.add_argument('--kg_dir',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
 spans=loadj(Path(a.candidate_dir)/'corpus_spans.jsonl'); preds=loadj(Path(a.inference_dir)/'ensemble_relation_predictions.jsonl'); nodes=loadj(Path(a.kg_dir)/'kg_nodes.jsonl'); edges=loadj(Path(a.kg_dir)/'kg_edges.jsonl'); fam=loadj(Path(a.kg_dir)/'hypothesis_families.jsonl')
 _require_locked_ids({x.get("paper_master_id") for x in spans+preds+nodes+edges})
 bad=({x.get('paper_master_id') for x in spans}|{x.get('paper_master_id') for x in preds}|{x.get('paper_master_id') for x in nodes}|{x.get('paper_master_id') for x in edges}) & LOCKED
 ids={x['id'] for x in nodes}; missing_refs=[(e['source'],e['target']) for e in edges if e['source'] not in ids or e['target'] not in ids]
 bad_support=[r['pair_id'] for r in preds if r.get('effective_relation_label')=='SUPPORTS' and r.get('supports_probability_mean',0)<0.5 and not r.get('explicit_relation')]
 bad_mean=[]
 for r in preds:
  vals=[r.get('supports_probability_seed42'),r.get('supports_probability_seed52'),r.get('supports_probability_seed62')]
  if all(v is not None for v in vals) and abs(r['supports_probability_mean']-sum(vals)/3)>2e-6: bad_mean.append(r['pair_id'])
 report={'status':'PASSED' if not bad and not missing_refs and not bad_support and not bad_mean else 'FAILED','locked_papers':sorted(x for x in bad if x),'missing_edge_node_refs':len(missing_refs),'support_below_frozen_threshold':len(bad_support),'ensemble_mean_mismatch':len(bad_mean),'families':len(fam),'multi_branch_candidates':sum(bool(x.get('multi_branch_candidate')) for x in fam),'competition_asserted_automatically':False,'locked_test_label_files_read':False}
 Path(a.output).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
 if report['status']!='PASSED': raise SystemExit(2)
if __name__=='__main__': main()

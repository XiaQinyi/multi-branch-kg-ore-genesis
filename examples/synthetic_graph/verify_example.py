#!/usr/bin/env python3
"""Validate a small Multi-KG CSV example. Standard-library-only; no model inference."""
import argparse, csv, json
from pathlib import Path
from collections import Counter

REQUIRED_NODE={'id','type','text','dimension','paper_master_id','title','deposit_id','deposit_name','claim_dimension','canonical_claim_text','competition_status','interpretation_status'}
REQUIRED_EDGE={'source','target','type','source_kind','paper_master_id','pair_id','supports_probability_mean','supports_probability_std','supports_seed_votes','ensemble_vote_tier','evidence_dimension','claim_dimension'}
VALID_NODE={'DEPOSIT','PAPER','HYPOTHESIS_FAMILY','HYPOTHESIS_BRANCH','CLAIM','EVIDENCE'}
VALID_RELS={'STUDIES':('PAPER','DEPOSIT'),'ASSERTS':('PAPER','CLAIM'),'REPORTS':('PAPER','EVIDENCE'),'HAS_BRANCH':('HYPOTHESIS_FAMILY','HYPOTHESIS_BRANCH'),'CENTRAL_CLAIM':('HYPOTHESIS_BRANCH','CLAIM'),'SUPPORTS':('EVIDENCE','CLAIM'),'OPPOSES':('EVIDENCE','CLAIM'),'QUALIFIES':('EVIDENCE','CLAIM')}

def load(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f)
        return list(r),set(r.fieldnames or [])

def verify(directory,expected_nodes=None,expected_edges=None):
    n,nfields=load(directory/'nodes.csv')
    e,efields=load(directory/'edges.csv')
    assert REQUIRED_NODE<=nfields, f'Missing node columns: {REQUIRED_NODE-nfields}'
    assert REQUIRED_EDGE<=efields, f'Missing edge columns: {REQUIRED_EDGE-efields}'
    byid={x['id']:x for x in n}
    assert len(n)==len(byid), 'Duplicate node IDs'
    assert all(x['type'] in VALID_NODE for x in n), 'Unexpected node type'
    assert all(x['source'] in byid and x['target'] in byid for x in e), 'Dangling relation'
    assert all(x['type'] in VALID_RELS for x in e), 'Unexpected relation type'
    for x in e:
        expected=VALID_RELS[x['type']]
        assert (byid[x['source']]['type'],byid[x['target']]['type'])==expected, f'Bad directed type for {x["type"]}'
    branches={x['id'] for x in n if x['type']=='HYPOTHESIS_BRANCH'}
    central=Counter(x['source'] for x in e if x['type']=='CENTRAL_CLAIM')
    family=Counter(x['target'] for x in e if x['type']=='HAS_BRANCH')
    assert all(central[b]==1 and family[b]==1 for b in branches), 'Each branch must belong to a family and have exactly one central claim'
    assert all(x['competition_status'] in ('','UNRESOLVED') for x in n if x['type']=='HYPOTHESIS_BRANCH'), 'Competition must not be asserted'
    if expected_nodes is not None: assert len(n)==expected_nodes,(len(n),expected_nodes)
    if expected_edges is not None: assert len(e)==expected_edges,(len(e),expected_edges)
    report={'status':'PASSED','nodes':len(n),'edges':len(e),'dangling_edges':0,'node_types':dict(Counter(x['type'] for x in n)),'edge_types':dict(Counter(x['type'] for x in e)),'unresolved_branches':len(branches)}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--graph-dir',type=Path,default=Path(__file__).parent)
    p.add_argument('--expected-nodes',type=int);p.add_argument('--expected-edges',type=int)
    a=p.parse_args();verify(a.graph_dir,a.expected_nodes,a.expected_edges)

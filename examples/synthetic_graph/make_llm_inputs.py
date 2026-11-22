#!/usr/bin/env python3
"""Build illustrative Flat-KG and Multi-KG inputs from the SAME CSV graph atoms.
Does not perform retrieval, semantic adjudication or any manuscript scoring.
"""
import argparse,csv,json
from pathlib import Path
from collections import defaultdict

def read_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser();p.add_argument('--graph-dir',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    nodes=read_csv(a.graph_dir/'nodes.csv'); edges=read_csv(a.graph_dir/'edges.csv')
    byid={n['id']:n for n in nodes}
    case_file=a.graph_dir/'cases.jsonl'
    cases=[json.loads(s) for s in case_file.read_text(encoding='utf-8').splitlines() if s.strip()]
    # Shared case scope is determined through the family -> branch -> claim hierarchy.
    fam_to_branch=defaultdict(list);branch_to_claim={};evidence_link=defaultdict(list); claim_to_paper={};evidence_to_paper={}
    for e in edges:
        if e['type']=='HAS_BRANCH':fam_to_branch[e['source']].append(e['target'])
        if e['type']=='CENTRAL_CLAIM':branch_to_claim[e['source']]=e['target']
        if e['type'] in ('SUPPORTS','OPPOSES','QUALIFIES'):evidence_link[e['target']].append((e['type'],e['source']))
        if e['type']=='ASSERTS': claim_to_paper[e['target']]=e['source']
        if e['type']=='REPORTS': evidence_to_paper[e['target']]=e['source']
    families=sorted(n['id'] for n in nodes if n['type']=='HYPOTHESIS_FAMILY')
    if len(cases)!=len(families):raise ValueError('cases.jsonl must have one entry for each hypothesis family')
    records={'flat_kg':[],'multibranch_kg':[]}
    checks=[]
    for i,(family,case) in enumerate(zip(families,cases),1):
        branches=sorted(fam_to_branch[family]);assert len(branches)>=2, 'Each demo family needs two branches'
        units=[];grouped=[]
        for branch in branches:
            claim=branch_to_claim[branch]
            claim_node=byid[claim]
            clabel=claim_node['text']
            claim_atom={'id':claim,'text':f'CLAIM {claim}: {clabel}'}
            current=[claim_atom]
            for etype,evid in sorted(evidence_link[claim]):
                current.append({'id':f'{evid}->{claim}:{etype}', 'text':f'{etype}: {evid}: {byid[evid]["text"]}'})
            grouped.append((branch,current));units.extend(current)
        # identical atom IDs/text in both conditions; Flat reverses order.
        flat_atoms=list(reversed(units))
        assert sorted((x['id'],x['text']) for x in flat_atoms)==sorted((x['id'],x['text']) for x in units)
        flat='UNGROUPED EVIDENCE ATOMS\n'+'\n'.join(x['text'] for x in flat_atoms)
        multi='BRANCH-GROUPED EVIDENCE ATOMS\n'+'\n'.join('BRANCH '+str(j)+' (competition unresolved)\n'+'\n'.join(x['text'] for x in atomset) for j,(_,atomset) in enumerate(grouped,1))
        prompt='Answer with valid JSON and cite evidence IDs. Treat all branches as hypotheses, not verified mutually exclusive interpretations.'
        for condition,context in (('flat_kg',flat),('multibranch_kg',multi)):
            records[condition].append({'benchmark_id':case['case_id'],'condition':condition.upper(),'question':(case.get('question') or ('What geological mechanisms could explain the observations for '+case.get('demonstration_topic','this example')+'?')),'context':context,'system_prompt':prompt,'user_prompt':(case.get('question') or ('What geological mechanisms could explain the observations for '+case.get('demonstration_topic','this example')+'?'))+'\n\n'+context})
        checks.append({'case_id':case['case_id'],'family_id':family,'branches':len(branches),'atoms_per_condition':len(units),'flat_multi_atom_parity':'EXACT'})
    for condition,rows in records.items():
        with (a.output_dir/(condition+'.jsonl')).open('w',encoding='utf-8') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    (a.output_dir/'parity_audit.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'PASS: {len(cases)} cases; flat and grouped inputs created; per-case atom parity EXACT')
if __name__=='__main__':main()

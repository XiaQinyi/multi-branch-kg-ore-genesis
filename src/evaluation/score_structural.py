#!/usr/bin/env python3
# Deterministic structural metrics only. Semantic branch coverage requires blinded human/fixed judge.
import argparse,json,re,collections
p=argparse.ArgumentParser();p.add_argument('--input',required=True);a=p.parse_args()
rows=[]
for line in open(a.input,encoding='utf-8'):
 r=json.loads(line); raw=r['raw_output'].strip(); raw=re.sub(r'^```(?:json)?\s*|\s*```$','',raw,flags=re.S)
 try:o=json.loads(raw)
 except: rows.append((r['benchmark_id'],r['condition'],0,0,0));continue
 ints=o.get('interpretations',[]) if isinstance(o.get('interpretations'),list) else []
 refs=[x for i in ints for x in (i.get('evidence_refs') or []) if isinstance(x,str)]
 halluc=sum(1 for x in refs if not re.fullmatch(r'[ST]\d{2}',x))
 rows.append((r['benchmark_id'],r['condition'],1,len(ints),halluc))
print('benchmark_id\tcondition\tjson_valid\tinterpretation_count\tmalformed_evidence_ref_count')
for x in rows:print('\t'.join(map(str,x)))

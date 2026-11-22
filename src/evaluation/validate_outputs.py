#!/usr/bin/env python3
import argparse,json,re
p=argparse.ArgumentParser();p.add_argument('--input',required=True);a=p.parse_args()
N=ok=0
for line in open(a.input,encoding='utf-8'):
 N+=1;r=json.loads(line);s=r['raw_output'].strip();
 try:
  # tolerate accidental fences
  s=re.sub(r'^```(?:json)?\s*|\s*```$','',s,flags=re.S)
  o=json.loads(s); good=isinstance(o.get('interpretations'),list) and 'answer' in o
  ok+=int(good)
  print(r['benchmark_id'],r['condition'],'VALID' if good else 'INVALID',len(o.get('interpretations',[])) if good else '')
 except Exception as e: print(r['benchmark_id'],r['condition'],'INVALID_JSON',str(e)[:120])
print('VALID_JSON',ok,'/',N)

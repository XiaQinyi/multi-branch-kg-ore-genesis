#!/usr/bin/env python3
import argparse,json,torch
from pathlib import Path
from transformers import AutoTokenizer,AutoModelForCausalLM
p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--input',required=True); p.add_argument('--output',required=True); p.add_argument('--max-new-tokens',type=int,default=900); p.add_argument('--temperature',type=float,default=0.0); p.add_argument('--seed',type=int,default=20260808); a=p.parse_args()
torch.manual_seed(a.seed)
tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True)
model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype='auto',device_map='auto',trust_remote_code=True).eval()
out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
with open(a.input,encoding='utf-8') as f, out.open('w',encoding='utf-8') as g:
  for line in f:
    r=json.loads(line); msgs=[{'role':'system','content':r['system_prompt']},{'role':'user','content':r['user_prompt']}]
    text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True) if hasattr(tok,'apply_chat_template') else r['system_prompt']+'\n'+r['user_prompt']
    x=tok(text,return_tensors='pt').to(model.device)
    kw={'max_new_tokens':a.max_new_tokens,'do_sample':a.temperature>0}
    if a.temperature>0: kw['temperature']=a.temperature
    with torch.no_grad(): y=model.generate(**x,**kw)
    ans=tok.decode(y[0][x['input_ids'].shape[1]:],skip_special_tokens=True)
    g.write(json.dumps({'benchmark_id':r['benchmark_id'],'condition':r['condition'],'raw_output':ans},ensure_ascii=False)+'\n'); g.flush()

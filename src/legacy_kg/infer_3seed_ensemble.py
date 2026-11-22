#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModel

DIMS=["AGE","SOURCE","STRUCTURE","PROCESS","CHEMISTRY","MINERALIZATION","SCALE"]

def load_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]

class M(nn.Module):
    def __init__(self,enc_dir,heads):
        super().__init__()
        self.encoder=AutoModel.from_pretrained(enc_dir,local_files_only=True,add_pooling_layer=False)
        h=self.encoder.config.hidden_size
        self.rel=nn.Linear(h,2); self.edim=nn.Linear(h,7); self.cdim=nn.Linear(h,7)
        st=torch.load(heads,map_location="cpu")
        self.rel.load_state_dict(st["relation_head"])
        self.edim.load_state_dict(st["evidence_dimension_head"])
        self.cdim.load_state_dict(st["claim_dimension_head"])
    def forward(self,**enc):
        z=self.encoder(**enc).last_hidden_state[:,0]
        return self.rel(z),self.edim(z),self.cdim(z)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--candidates",required=True)
    ap.add_argument("--spans",required=True)
    ap.add_argument("--model_root",required=True)
    ap.add_argument("--output_dir",required=True)
    ap.add_argument("--batch_size",type=int,default=64)
    ap.add_argument("--max_length",type=int,default=512)
    a=ap.parse_args()
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    rows=load_jsonl(a.candidates)
    spans=load_jsonl(a.spans)
    LOCKED={item.strip() for item in __import__('os').environ.get('MULTIKG_LOCKED_PAPER_IDS','').split(',') if item.strip()}
    if not LOCKED and any(pid and not str(pid).startswith('DEMO-PAPER-') for pid in ({r.get('paper_master_id') for r in rows}|{s.get('paper_master_id') for s in spans})):
        raise RuntimeError('Set MULTIKG_LOCKED_PAPER_IDS before processing non-demo inputs')
    bad=({r.get("paper_master_id") for r in rows}|{s.get("paper_master_id") for s in spans}) & LOCKED
    if bad: raise ValueError(f"Locked papers are forbidden in KG v1 inference: {sorted(bad)}")
    n=len(rows)
    if not n: raise ValueError("no candidates")
    seeds=[42,52,62]
    rel_seed=np.zeros((len(seeds),n),dtype=np.float32)
    e_seed=np.zeros((len(seeds),n,7),dtype=np.float32)
    c_seed=np.zeros((len(seeds),n,7),dtype=np.float32)

    first_enc=Path(a.model_root)/"seed_42/final_model/encoder"
    tok=AutoTokenizer.from_pretrained(first_enc,local_files_only=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    texts=[r["model_text"] for r in rows]

    for si,seed in enumerate(seeds):
        base=Path(a.model_root)/f"seed_{seed}/final_model"
        enc_dir=base/"encoder"; heads=base/"heads.pt"
        if not enc_dir.joinpath("config.json").is_file() or not heads.is_file():
            raise FileNotFoundError(f"missing final seed model {seed}: {base}")
        model=M(enc_dir,heads).to(device).eval()
        with torch.inference_mode():
            for start in range(0,n,a.batch_size):
                batch=texts[start:start+a.batch_size]
                enc=tok(batch,padding=True,truncation=True,max_length=a.max_length,return_tensors="pt")
                enc={k:v.to(device) for k,v in enc.items()}
                r,e,c=model(**enc)
                rel_seed[si,start:start+len(batch)]=F.softmax(r,dim=-1)[:,1].float().cpu().numpy()
                e_seed[si,start:start+len(batch)]=F.softmax(e,dim=-1).float().cpu().numpy()
                c_seed[si,start:start+len(batch)]=F.softmax(c,dim=-1).float().cpu().numpy()
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(json.dumps({"seed_done":seed,"pairs":n},ensure_ascii=False))

    rel_mean=rel_seed.mean(0); rel_std=rel_seed.std(0)
    e_mean=e_seed.mean(0); c_mean=c_seed.mean(0)

    # Pair predictions and stable per-span dimension aggregation.
    span_sum=defaultdict(lambda:np.zeros(7,dtype=np.float64))
    span_count=defaultdict(int)
    pred_path=out/"ensemble_relation_predictions.jsonl"
    with pred_path.open("w",encoding="utf-8") as f:
        for i,r in enumerate(rows):
            model_lab="SUPPORTS" if rel_mean[i]>=0.5 else "NO_RELATION"
            explicit=r.get("explicit_relation","")
            effective=explicit if explicit in {"SUPPORTS","QUALIFIES","OPPOSES"} else model_lab
            source=r.get("explicit_relation_source") if explicit else ("MODEL" if model_lab=="SUPPORTS" else "MODEL_NO_RELATION")
            ed=DIMS[int(e_mean[i].argmax())]; cd=DIMS[int(c_mean[i].argmax())]
            seed_votes=int(sum(rel_seed[:,i]>=0.5))
            x=dict(r)
            x.update({
                "supports_probability_mean":float(rel_mean[i]),
                "supports_probability_std":float(rel_std[i]),
                "supports_probability_seed42":float(rel_seed[0,i]),
                "supports_probability_seed52":float(rel_seed[1,i]),
                "supports_probability_seed62":float(rel_seed[2,i]),
                "supports_seed_votes":seed_votes,
                "model_relation_label":model_lab,
                "effective_relation_label":effective,
                "effective_relation_source":source,
                "evidence_dimension":ed,
                "evidence_dimension_probabilities":{DIMS[j]:float(e_mean[i,j]) for j in range(7)},
                "claim_dimension":cd,
                "claim_dimension_probabilities":{DIMS[j]:float(c_mean[i,j]) for j in range(7)},
                "dimension_pair":f"{ed}->{cd}",
                "ensemble_policy":"3SEED_EQUAL_WEIGHT_MEAN",
                "relation_threshold":0.5
            })
            f.write(json.dumps(x,ensure_ascii=False)+"\n")
            span_sum[r["evidence_uid"]]+=e_mean[i]; span_count[r["evidence_uid"]]+=1
            span_sum[r["claim_uid"]]+=c_mean[i]; span_count[r["claim_uid"]]+=1

    span_meta={s["span_uid"]:s for s in spans}
    with (out/"span_dimension_ensemble.jsonl").open("w",encoding="utf-8") as f:
        for uid,s in span_meta.items():
            if span_count[uid]:
                p=(span_sum[uid]/span_count[uid]); p=p/p.sum()
                lab=DIMS[int(p.argmax())]
                source="MODEL_3SEED_PAIR_CONTEXT_MEAN"
            else:
                p=np.ones(7)/7; lab="UNRESOLVED"; source="NO_PAIR_CONTEXT"
            x=dict(s)
            x.update({
                "dimension":lab,
                "dimension_probabilities":{DIMS[j]:float(p[j]) for j in range(7)},
                "pair_context_count":span_count[uid],
                "dimension_source":source
            })
            f.write(json.dumps(x,ensure_ascii=False)+"\n")

    summary={
        "status":"PASSED",
        "pairs":n,
        "model_supports_at_0.5":int((rel_mean>=0.5).sum()),
        "model_no_relation_at_0.5":int((rel_mean<0.5).sum()),
        "explicit_relations_overriding_binary":sum(bool(r.get("explicit_relation")) for r in rows),
        "three_seed_unanimous_supports":int(((rel_seed>=0.5).sum(0)==3).sum()),
        "three_seed_unanimous_no_relation":int(((rel_seed>=0.5).sum(0)==0).sum()),
        "locked_test_label_files_read":False
    }
    (out/"ensemble_inference_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__":
    main()

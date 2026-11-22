#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json,re
from collections import Counter,defaultdict
from pathlib import Path

import os
LOCKED = {item.strip() for item in os.environ.get("MULTIKG_LOCKED_PAPER_IDS", "").split(",") if item.strip()}
def _require_locked_ids(paper_ids):
    if not LOCKED and any(pid and not str(pid).startswith("DEMO-PAPER-") for pid in paper_ids):
        raise RuntimeError("Set MULTIKG_LOCKED_PAPER_IDS before processing non-demo paper IDs; never publish this private exclusion list")


def loadj(p): return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]
def h(s): return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]
def norm(s): return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+","",(s or "").lower())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--predictions",required=True); ap.add_argument("--span_dimensions",required=True); ap.add_argument("--output_dir",required=True); a=ap.parse_args()
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    preds=loadj(a.predictions); spans=loadj(a.span_dimensions)
    _require_locked_ids({s.get("paper_master_id") for s in spans}|{r.get("paper_master_id") for r in preds})
    if ({s["paper_master_id"] for s in spans}|{r["paper_master_id"] for r in preds}) & LOCKED: raise RuntimeError("Locked paper leaked into KG build")

    nodes={}; edges=[]; paper_deposit=set(); family_branch_edge=set(); claim_to_branch={}; branch_claims=defaultdict(list); branch_rels=defaultdict(list); family_branches=defaultdict(set)
    def node(i,t,**kw):
        if i not in nodes: nodes[i]={"id":i,"type":t,**kw}
    for s in spans:
        pid=s["paper_master_id"]; dep=s.get("deposit_id") or "DEPOSIT_UNASSIGNED"; pap=f"PAPER::{pid}"; did=f"DEPOSIT::{dep}"
        node(pap,"PAPER",paper_master_id=pid,title=s.get("title",""),split=s.get("split",""),source_role=s.get("source_role",""),scientific_analysis_eligible=s.get("scientific_analysis_eligible",True))
        node(did,"DEPOSIT",deposit_id=dep,deposit_name=s.get("deposit_name",""))
        if (pap,did) not in paper_deposit: edges.append({"source":pap,"target":did,"type":"STUDIES","source_kind":"PAPER_MASTER_METADATA"}); paper_deposit.add((pap,did))
        sid=("CLAIM::" if s["role"]=="CLAIM" else "EVIDENCE::")+s["span_uid"]
        node(sid,s["role"],span_uid=s["span_uid"],text=s.get("text",""),dimension=s.get("dimension","UNRESOLVED"),paper_master_id=pid,deposit_id=dep,source_unit_id=s.get("source_unit_id",""),source_kind=s.get("source_kind",""))
        edges.append({"source":pap,"target":sid,"type":"ASSERTS" if s["role"]=="CLAIM" else "REPORTS","source_kind":"CORPUS"})
        if s["role"]=="CLAIM" and dep!="DEPOSIT_UNASSIGNED" and s.get("dimension") not in {"",None,"UNRESOLVED"}:
            dim=s["dimension"]; fam=f"FAMILY::{dep}::{dim}"; br=f"BRANCH::{h(fam+'|'+norm(s.get('text','')))}"
            node(fam,"HYPOTHESIS_FAMILY",deposit_id=dep,claim_dimension=dim,interpretation_status="PARALLEL_OR_COMPETING_UNRESOLVED")
            node(br,"HYPOTHESIS_BRANCH",deposit_id=dep,claim_dimension=dim,canonical_claim_text=s.get("text",""),canonicalization="EXACT_NORMALIZED_TEXT_ONLY",competition_status="UNRESOLVED")
            family_branches[fam].add(br); branch_claims[br].append(sid); claim_to_branch[sid]=br
            if (fam,br) not in family_branch_edge:
                edges.append({"source":fam,"target":br,"type":"HAS_BRANCH","source_kind":"DERIVED_STRUCTURE"}); family_branch_edge.add((fam,br))
            edges.append({"source":br,"target":sid,"type":"CENTRAL_CLAIM","source_kind":"DERIVED_STRUCTURE"})

    relation_edges=[]
    for r in preds:
        lab=r.get("effective_relation_label")
        if lab not in {"SUPPORTS","QUALIFIES","OPPOSES"}: continue
        e="EVIDENCE::"+r["evidence_uid"]; c="CLAIM::"+r["claim_uid"]
        tier="UNANIMOUS" if r.get("supports_seed_votes")==3 else ("MAJORITY" if r.get("supports_seed_votes")==2 else "ENSEMBLE_MEAN_ONLY")
        ed={"source":e,"target":c,"type":lab,"source_kind":r.get("effective_relation_source",""),"paper_master_id":r["paper_master_id"],"pair_id":r["pair_id"],
            "supports_probability_mean":r.get("supports_probability_mean"),"supports_probability_std":r.get("supports_probability_std"),"supports_seed_votes":r.get("supports_seed_votes"),
            "ensemble_vote_tier":tier,"evidence_dimension":r.get("evidence_dimension"),"claim_dimension":r.get("claim_dimension")}
        edges.append(ed); relation_edges.append(ed)
        if c in claim_to_branch: branch_rels[claim_to_branch[c]].append(ed)

    families=[]
    for fam,bs in sorted(family_branches.items()):
        families.append({"family_id":fam,"deposit_id":nodes[fam]["deposit_id"],"claim_dimension":nodes[fam]["claim_dimension"],"branch_count":len(bs),"multi_branch_candidate":len(bs)>=2,
                         "status":"PARALLEL_OR_COMPETING_UNRESOLVED" if len(bs)>=2 else "SINGLE_BRANCH","branch_ids":sorted(bs)})
    branches=[]
    for br,claims in sorted(branch_claims.items()):
        rs=branch_rels.get(br,[]); supp=[x for x in rs if x["type"]=="SUPPORTS"]
        branches.append({"branch_id":br,"deposit_id":nodes[br]["deposit_id"],"claim_dimension":nodes[br]["claim_dimension"],"canonical_claim_text":nodes[br]["canonical_claim_text"],"claim_instance_count":len(claims),
                         "support_count":len(supp),"qualifies_count":sum(x["type"]=="QUALIFIES" for x in rs),"opposes_count":sum(x["type"]=="OPPOSES" for x in rs),
                         "mean_support_probability":(sum(x["supports_probability_mean"] for x in supp)/len(supp) if supp else None),"claim_node_ids":claims,"competition_status":"UNRESOLVED"})

    for name,rows in [("kg_nodes.jsonl",nodes.values()),("kg_edges.jsonl",edges),("hypothesis_branches.jsonl",branches),("hypothesis_families.jsonl",families)]:
        (out/name).write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
    nf=["id","type","text","dimension","paper_master_id","title","deposit_id","deposit_name","claim_dimension","canonical_claim_text","competition_status","interpretation_status"]
    with (out/"nodes.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=nf); w.writeheader(); [w.writerow({k:x.get(k,"") for k in nf}) for x in nodes.values()]
    ef=["source","target","type","source_kind","paper_master_id","pair_id","supports_probability_mean","supports_probability_std","supports_seed_votes","ensemble_vote_tier","evidence_dimension","claim_dimension"]
    with (out/"edges.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=ef); w.writeheader(); [w.writerow({k:x.get(k,"") for k in ef}) for x in edges]
    summary={"status":"PASSED","nodes":len(nodes),"edges":len(edges),"papers":sum(x["type"]=="PAPER" for x in nodes.values()),"deposits":sum(x["type"]=="DEPOSIT" for x in nodes.values()),
             "relation_edges":len(relation_edges),"relation_edge_types":dict(Counter(x["type"] for x in relation_edges)),"relation_edge_sources":dict(Counter(x["source_kind"] for x in relation_edges)),
             "branches":len(branches),"families":len(families),"multi_branch_candidate_families":sum(x["multi_branch_candidate"] for x in families),
             "semantic_branch_merging_performed":False,"competition_asserted_automatically":False,"locked_test_label_files_read":False,"locked_papers_present":False}
    (out/"kg_build_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__": main()

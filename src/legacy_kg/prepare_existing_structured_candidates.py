#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,io,json,re,zipfile
from collections import Counter,defaultdict
from pathlib import Path

import os
LOCKED = {item.strip() for item in os.environ.get("MULTIKG_LOCKED_PAPER_IDS", "").split(",") if item.strip()}
def _require_locked_ids(paper_ids):
    if not LOCKED and any(pid and not str(pid).startswith("DEMO-PAPER-") for pid in paper_ids):
        raise RuntimeError("Set MULTIKG_LOCKED_PAPER_IDS before processing non-demo paper IDs; never publish this private exclusion list")

RARE={"QUALIFIES","OPPOSES"}
CE={"CLAIM","EVIDENCE"}

def norm(s): return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+","",(s or "").lower())
def h16(s): return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]
def b(v): return str(v).strip().lower() in {"1","true","yes","y"}
def first(r,*keys,default=""):
    for k in keys:
        v=r.get(k)
        if v not in (None,""): return v
    return default

def read_zip_member(z,suffix):
    ms=[n for n in z.namelist() if n.endswith(suffix)]
    if len(ms)!=1: raise ValueError(f"expected exactly one *{suffix}, got {ms}")
    return z.read(ms[0]).decode("utf-8-sig")

def extract_result(task):
    for k in ("annotations","predictions"):
        arr=task.get(k)
        if isinstance(arr,list) and arr:
            for x in arr:
                if isinstance(x,dict) and isinstance(x.get("result"),list): return x["result"]
    return []

def parse_ls(task):
    spans={}; rels={}
    for x in extract_result(task):
        if x.get("type")!="labels": continue
        v=x.get("value") or {}; labs=v.get("labels") or []
        if not labs or labs[0] not in CE: continue
        sid=str(x.get("id") or h16(json.dumps(v,ensure_ascii=False,sort_keys=True)))
        spans[sid]={"id":sid,"label":labs[0],"start":int(v.get("start",-1)),"end":int(v.get("end",-1)),"text":v.get("text","")}
    for x in extract_result(task):
        if x.get("type")!="relation": continue
        labs=x.get("labels") or []
        if not labs: continue
        lab=labs[0]; a=spans.get(str(x.get("from_id"))); c=spans.get(str(x.get("to_id")))
        if not a or not c: continue
        if a["label"]=="EVIDENCE" and c["label"]=="CLAIM": key=(a["id"],c["id"])
        elif c["label"]=="EVIDENCE" and a["label"]=="CLAIM": key=(c["id"],a["id"])
        else: continue
        rels[key]=lab
    return spans,rels

def gap(a,b):
    if a.get("end",-1)>=0 and b.get("start",-1)>=0 and a["end"] < b["start"]: return b["start"]-a["end"]
    if b.get("end",-1)>=0 and a.get("start",-1)>=0 and b["end"] < a["start"]: return a["start"]-b["end"]
    return 0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--master_zip",required=True)
    ap.add_argument("--output_dir",required=True)
    ap.add_argument("--weak_pairs",default="AUTO")
    ap.add_argument("--char_gap_max",type=int,default=500)
    a=ap.parse_args()
    if not LOCKED: raise RuntimeError("Set MULTIKG_LOCKED_PAPER_IDS before preparing non-demo real research candidates")
    root=Path(a.root); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    master_zip=Path(a.master_zip)
    if any(x in str(master_zip) for x in ["LockedTest","FROZEN_LABELS","locked_test"]):
        raise ValueError("Locked Test path is forbidden")

    # Prefer the reviewed split table in the active project; fall back to the frozen Paper Master zip.
    reviewed=root/"data/master/final_paper_master_with_reviewed_splits.csv"
    with zipfile.ZipFile(master_zip) as z:
        if reviewed.is_file():
            master_rows=list(csv.DictReader(reviewed.open(encoding="utf-8-sig")))
        else:
            master_rows=list(csv.DictReader(io.StringIO(read_zip_member(z,"04_Splits/final_paper_master_with_splits.csv"))))
        gold_tasks=json.loads(read_zip_member(z,"03_Gold/gold_fulltext_80_LabelStudio.json"))
        legacy=[json.loads(x) for x in read_zip_member(z,"05_Legacy/legacy_segments_with_paper_master.jsonl").splitlines() if x.strip()]

    master={r["paper_master_id"]:r for r in master_rows}
    if len(master)!=742: raise ValueError(f"Paper Master expected 742 rows, got {len(master)}")
    mid2pid={r.get("current_master_doc_id",""):r["paper_master_id"] for r in master_rows if r.get("current_master_doc_id")}
    oid2pid={r.get("current_original_doc_id",""):r["paper_master_id"] for r in master_rows if r.get("current_original_doc_id")}
    title2pid=defaultdict(list)
    for r in master_rows: title2pid[norm(r.get("canonical_title",""))].append(r["paper_master_id"])

    def resolve_pid(r):
        for k in ("paper_master_id","paper_id"):
            v=str(r.get(k) or "")
            if v in master: return v
        for k in ("master_doc_id","document_id","doc_id"):
            v=str(r.get(k) or "")
            if v in mid2pid: return mid2pid[v]
            if v in oid2pid: return oid2pid[v]
        t=norm(first(r,"canonical_title","title","paper_title"))
        if t and len(title2pid.get(t,[]))==1: return title2pid[t][0]
        return ""

    spans_out={}; cand_out={}; represented=set(); source_papers=defaultdict(set); source_units=Counter(); skipped=Counter()

    def meta(pid):
        r=master[pid]
        return {
            "paper_master_id":pid,"title":r.get("canonical_title",""),"deposit_id":r.get("deposit_id",""),
            "deposit_name":r.get("deposit_name",""),"primary_rq":r.get("primary_rq",""),"split":r.get("split",""),
            "source_role":r.get("source_role",r.get("source_role_candidate","")),
            "scientific_analysis_eligible":b(r.get("scientific_analysis_eligible","True")),
            "source_availability":r.get("source_availability","")}

    def add_span(pid,unit,source_kind,s):
        m=meta(pid); sid=str(s.get("id") or h16(f"{pid}|{unit}|{s.get('label')}|{s.get('start')}|{s.get('end')}|{s.get('text','')}"))
        uid=f"{pid}|{unit}|{sid}"
        x={**m,"span_uid":uid,"source_unit_id":unit,"source_kind":source_kind,"role":s["label"],"span_id":sid,
           "start":int(s.get("start",-1) if s.get("start",-1) not in (None,"") else -1),
           "end":int(s.get("end",-1) if s.get("end",-1) not in (None,"") else -1),"text":s.get("text","")}
        spans_out.setdefault(uid,x)
        return uid

    def add_pair(pid,unit,source_kind,e,c,explicit="",explicit_source=""):
        eu=add_span(pid,unit,source_kind,e); cu=add_span(pid,unit,source_kind,c)
        key=f"{pid}|{unit}|{e.get('id')}|{c.get('id')}|{e.get('start')}|{c.get('start')}|{e.get('text')}|{c.get('text')}"
        pair_id="PAIR-"+h16(key); m=meta(pid)
        row={**m,"pair_id":pair_id,"source_unit_id":unit,"source_kind":source_kind,
             "evidence_uid":eu,"evidence_id":e.get("id",""),"evidence_text":e.get("text",""),"evidence_start":int(e.get("start",-1) if e.get("start",-1) not in (None,"") else -1),"evidence_end":int(e.get("end",-1) if e.get("end",-1) not in (None,"") else -1),
             "claim_uid":cu,"claim_id":c.get("id",""),"claim_text":c.get("text",""),"claim_start":int(c.get("start",-1) if c.get("start",-1) not in (None,"") else -1),"claim_end":int(c.get("end",-1) if c.get("end",-1) not in (None,"") else -1),
             "char_gap":gap(e,c),"explicit_relation":explicit if explicit in RARE else "",
             "explicit_relation_source":explicit_source if explicit in RARE else "",
             "model_text":f"[unused1]{e.get('text','')}[unused2] [unused3]{c.get('text','')}[unused4]"}
        cand_out.setdefault(pair_id,row); represented.add(pid); source_papers[source_kind].add(pid)

    # A) Gold current fulltext: trusted spans, preserve only explicit rare relation types.
    gold_pids=set()
    for t in gold_tasks:
        data=t.get("data") or {}; pid=mid2pid.get(data.get("master_doc_id",""),"")
        if not pid or pid in LOCKED: continue
        gold_pids.add(pid); source_units["CURRENT_GOLD_FULLTEXT"]+=1
        spans,rels=parse_ls(t); unit=data.get("master_doc_id") or pid
        source_papers["CURRENT_GOLD_FULLTEXT"].add(pid); represented.add(pid)
        for sx in spans.values(): add_span(pid,unit,"CURRENT_GOLD_FULLTEXT",sx)
        es=[x for x in spans.values() if x["label"]=="EVIDENCE"]; cs=[x for x in spans.values() if x["label"]=="CLAIM"]
        pairs={(e["id"],c["id"]) for e in es for c in cs if gap(e,c)<=a.char_gap_max}
        pairs |= {k for k,v in rels.items() if v in RARE}
        for eid,cid in pairs:
            add_pair(pid,unit,"CURRENT_GOLD_FULLTEXT",spans[eid],spans[cid],rels.get((eid,cid),""),"GOLD_EXPLICIT_RARE")
    if len(gold_pids)!=80: raise ValueError(f"Expected 80 Gold current papers, got {len(gold_pids)}")

    # B) Weak current fulltext: reuse existing model-ready candidate rows; never reuse weak binary labels.
    weak_path=(root/"data/relation_pairs/weak_train.jsonl") if a.weak_pairs=="AUTO" else Path(a.weak_pairs)
    weak_rows=[]
    if weak_path.is_file(): weak_rows=[json.loads(x) for x in weak_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    else: raise FileNotFoundError(f"weak relation-pair file not found: {weak_path}")
    weak_unresolved=0
    for r in weak_rows:
        pid=resolve_pid(r)
        if not pid: weak_unresolved+=1; continue
        if pid in LOCKED or pid in gold_pids: continue
        if master[pid].get("source_availability") not in {"FULL_TEXT","FULL_TEXT_PLUS_LEGACY_SEGMENTS"}: continue
        et=str(first(r,"evidence_text","head_text","source_text")); ct=str(first(r,"claim_text","tail_text","target_text"))
        if not et or not ct: skipped["weak_missing_text"]+=1; continue
        unit=str(first(r,"source_unit_id","window_id","segment_id","document_id","master_doc_id",default=master[pid].get("current_master_doc_id") or pid))
        es=int(first(r,"evidence_start","evidence_char_start","head_start",default=-1)); ee=int(first(r,"evidence_end","evidence_char_end","head_end",default=-1))
        cs=int(first(r,"claim_start","claim_char_start","tail_start",default=-1)); ce=int(first(r,"claim_end","claim_char_end","tail_end",default=-1))
        eid=str(first(r,"evidence_id","evidence_span_id","head_id",default="E"+h16(f"{et}|{es}|{ee}")))
        cid=str(first(r,"claim_id","claim_span_id","tail_id",default="C"+h16(f"{ct}|{cs}|{ce}")))
        e={"id":eid,"label":"EVIDENCE","start":es,"end":ee,"text":et}; c={"id":cid,"label":"CLAIM","start":cs,"end":ce,"text":ct}
        add_pair(pid,unit,"CURRENT_WEAK_FULLTEXT",e,c)
    weak_pids=source_papers["CURRENT_WEAK_FULLTEXT"]

    # C) Legacy segments: all SEGMENT_ONLY papers, plus FULL_TEXT_PLUS_LEGACY fallback only when current structured spans are absent.
    current_structured=set(gold_pids)|set(weak_pids)
    for t in legacy:
        pid=t.get("paper_master_id","")
        if pid not in master or pid in LOCKED: continue
        avail=master[pid].get("source_availability","")
        if avail=="SEGMENT_ONLY": source_kind="LEGACY_SEGMENT"
        elif avail=="FULL_TEXT_PLUS_LEGACY_SEGMENTS" and pid not in current_structured: source_kind="LEGACY_FALLBACK_FOR_MISSING_CURRENT_SPANS"
        else: continue
        unit=str(t.get("segment_id") or t.get("task_id") or t.get("document_id") or h16(t.get("text","")))
        source_units[source_kind]+=1
        spans={str(x["id"]):{"id":str(x["id"]),"label":x["label"],"start":int(x.get("start",-1)),"end":int(x.get("end",-1)),"text":x.get("text","")} for x in t.get("spans",[]) if x.get("label") in CE and x.get("id")}
        if spans:
            source_papers[source_kind].add(pid); represented.add(pid)
            for sx in spans.values(): add_span(pid,unit,source_kind,sx)
        rels={}
        for rr in t.get("relations",[]):
            lab=rr.get("label",""); a1=spans.get(str(rr.get("head"))); b1=spans.get(str(rr.get("tail")))
            if lab not in RARE or not a1 or not b1: continue
            if a1["label"]=="EVIDENCE" and b1["label"]=="CLAIM": rels[(a1["id"],b1["id"])]=lab
            elif b1["label"]=="EVIDENCE" and a1["label"]=="CLAIM": rels[(b1["id"],a1["id"])]=lab
        es=[x for x in spans.values() if x["label"]=="EVIDENCE"]; cs=[x for x in spans.values() if x["label"]=="CLAIM"]
        pairs={(e["id"],c["id"]) for e in es for c in cs if gap(e,c)<=a.char_gap_max}; pairs |= set(rels)
        for eid,cid in pairs: add_pair(pid,unit,source_kind,spans[eid],spans[cid],rels.get((eid,cid),""),"LEGACY_EXPLICIT_RARE")

    # Final locked guard.
    bad={x["paper_master_id"] for x in spans_out.values()} & LOCKED
    if bad: raise RuntimeError(f"Locked papers leaked into KG candidates: {sorted(bad)}")

    spans_path=out/"corpus_spans.jsonl"; cand_path=out/"relation_candidates.jsonl"
    spans_path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in spans_out.values()),encoding="utf-8")
    cand_path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in cand_out.values()),encoding="utf-8")
    represented={x["paper_master_id"] for x in spans_out.values()}; skipped_papers=sorted(set(master)-represented-LOCKED)
    summary={
        "status":"PASSED","paper_master_rows":len(master),"represented_papers":len(represented),"coverage_fraction":len(represented)/len(master),
        "locked_papers_excluded":sorted(LOCKED),"locked_test_label_files_read":False,
        "spans":len(spans_out),"candidates":len(cand_out),"explicit_rare_candidates":sum(bool(x.get("explicit_relation")) for x in cand_out.values()),
        "paper_counts_by_source":{k:len(v) for k,v in source_papers.items()},"source_unit_counts":dict(source_units),
        "gold_current_papers":len(gold_pids),"weak_current_papers":len(weak_pids),"weak_rows":len(weak_rows),"weak_unresolved_rows":weak_unresolved,
        "skipped_paper_count":len(skipped_papers),"skipped_papers":skipped_papers,"candidate_char_gap_max":a.char_gap_max,
        "binary_gold_labels_reused_for_edges":False,"weak_binary_labels_reused_for_edges":False,
        "rare_relation_policy":"Preserve explicit QUALIFIES/OPPOSES only; binary SUPPORTS/NO_RELATION comes from frozen final ensemble."
    }
    (out/"candidate_generation_audit.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__": main()

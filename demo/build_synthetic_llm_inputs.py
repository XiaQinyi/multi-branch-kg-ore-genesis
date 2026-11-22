#!/usr/bin/env python3
"""Make two illustrative input-format records using ONLY fictional example text.
Does not reproduce the paper's retrieval/ranking or claimed numeric results.
"""
import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', default=str(HERE / 'generated'))
    args = ap.parse_args()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = json.loads((HERE / 'data/fictional_branch_interpretations.json').read_text(encoding='utf-8'))
    a, b = source['interpretations']
    question = source['question']
    system = ('Respond as valid JSON with keys "answer" and "interpretations". Each interpretation '
              'should include a "summary" and an "evidence_refs" list. This is fictional teaching data.')
    common = [f"[{i['evidence_id']}] {i['claim']} Evidence: {i['evidence']}" for i in (a, b)]
    contexts = {
      'flat_kg': 'UNGROUPED RECORDS\n' + '\n'.join(reversed(common)),
      'multibranch_kg': ('BRANCH A: ' + a['summary'] + '\n' + common[0] + '\n'
                         'BRANCH B: ' + b['summary'] + '\n' + common[1] + '\n'
                         'RELATION: COMPETING (illustration, not inferred)'),
    }
    for condition, ctx in contexts.items():
        record = {'benchmark_id':source['question_id'], 'condition':condition,
                  'system_prompt':system,'user_prompt':f'{question}\n\n{ctx}'}
        (out/(condition+'.jsonl')).write_text(json.dumps(record,ensure_ascii=False)+'\n',encoding='utf-8')
    print('Created synthetic Flat-KG and Multi-KG input-format examples in',out)
if __name__ == '__main__': main()

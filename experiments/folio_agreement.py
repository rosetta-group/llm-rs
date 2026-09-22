"""Score two independently completed human review forms; refuse pending templates.

    python -m experiments.folio_agreement reviewer-A.json reviewer-B.json --out agreement.json

Metadata are declarations by reviewers, not proof of their independence. No scores
can be obtained from the supplied unfilled development templates.
"""
import argparse
import json
from pathlib import Path

from voynich.folio_objects import DOMAINS, PATTERNS


def binary_agreement(a,b):
    pairs=[(x,y) for x,y in zip(a,b) if x is not None and y is not None]
    n=len(pairs)
    if not n: return dict(n=0,unassessable=len(a),agreement=None,kappa=None,positive_agreement=None)
    yes_a=sum(x for x,y in pairs); yes_b=sum(y for x,y in pairs)
    both=sum(x and y for x,y in pairs); observed=sum(x==y for x,y in pairs)/n
    expected=(yes_a*yes_b+(n-yes_a)*(n-yes_b))/n**2
    return dict(n=n,unassessable=len(a)-n,agreement=observed,kappa=(observed-expected)/(1-expected) if expected<1 else None,
                positive_agreement=2*both/(yes_a+yes_b) if yes_a+yes_b else None,
                positives_a=yes_a,positives_b=yes_b)


def compare(a,b):
    for form in (a,b):
        if form['status']!='complete' or form['annotator_type']!='human' or not form['annotator_id'] or not all(form[k] is True for k in ('independent','text_masked','mask_reviewed')):
            raise ValueError('Require complete, independent human forms with reviewed text masking')
        mask=form.get('mask_manifest_sha256','')
        if not isinstance(mask,str) or len(mask)!=64 or any(c not in '0123456789abcdef' for c in mask):
            raise ValueError('Require mask manifest hash')
        ids=[r['panel_id'] for r in form['panels']]
        if not ids or len(ids)!=len(set(ids)): raise ValueError('Duplicate/empty panel set')
        for row in form['panels']:
            for block,keys in [('domains',DOMAINS),('patterns',PATTERNS)]:
                if set(row[block])!=set(keys): raise ValueError('Incomplete rubric fields')
                if any(v is not None and type(v) is not bool for v in row[block].values()): raise ValueError('Expected boolean or null')
                if any(v is None for v in row[block].values()) and not row.get('unassessable_reason'):
                    raise ValueError('Explain unassessable fields; do not leave pending blanks')
    if a['annotator_id']==b['annotator_id']: raise ValueError('Annotators must differ')
    if a['analysis_id']!=b['analysis_id'] or a['mask_manifest_sha256']!=b['mask_manifest_sha256']: raise ValueError('Different protocol or masks')
    aa={r['panel_id']:r for r in a['panels']}; bb={r['panel_id']:r for r in b['panels']}
    if set(aa)!=set(bb): raise ValueError('Panel sets differ')
    ids=sorted(aa)
    return dict(analysis_id=a['analysis_id'],panels=len(ids),mask_manifest_sha256=a['mask_manifest_sha256'],
                observer_declarations_only=True,
                scores={block:{k:binary_agreement([aa[i][block][k] for i in ids],[bb[i][block][k] for i in ids]) for k in keys}
                        for block,keys in [('domains',DOMAINS),('patterns',PATTERNS)]})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('first',type=Path);p.add_argument('second',type=Path);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists(): raise FileExistsError('Do not overwrite an agreement record')
    result=compare(json.loads(args.first.read_text()),json.loads(args.second.read_text()))
    args.out.write_text(json.dumps(result,indent=2)+'\n')

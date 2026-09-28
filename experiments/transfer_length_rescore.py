"""Post-hoc: rescore released rounds with one transfer length code per passage (`voynich.rejection_v3`).

python -m experiments.transfer_length_rescore

No fit or decode is repeated: the archived transfer records hold the decoded runs, separated by
'?' at every unreadable token. Sealed outcomes stay as recorded; this measures what the corrected
score would have decided, as development evidence for the next frozen rule.
"""
import json
from pathlib import Path
import time

from experiments import key_recovery_confirmation_v2 as v2c
from experiments import key_recovery_v2_development as v2d
from experiments.rejection_transfer_v2 import read, seal
from voynich.description_length import CharacterPrior
from voynich.rejection_development import decide_transfer
from voynich.rejection_v3 import rescore

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/transfer-length-rescore'
ROUND_ONE = ROOT / 'artifacts/key-recovery-confirmation'
PRIOR_FREEZE = ROOT / 'experiments/language-expansion/freeze.json'


def rounds():
    old = read(PRIOR_FREEZE)
    old_paths = {m: ROOT / 'artifacts/language-expansion/priors' / f'{m}.npz' for m in old['models']}
    new_entropy = dict(old['entropy'], **{m: r['entropy'] for m, r in read(v2d.OUT / 'priors.json')['models'].items()})
    new_paths = dict(old_paths, **{m: v2d.STATE / 'priors' / f'{m}.npz' for m in v2d.NEW.values()})

    def round_one(ident, model):
        return (read(ROUND_ONE / 'fit' / f'{ident}-{model}.json')['result'],
                read(ROUND_ONE / 'transfer' / f'{ident}-{model}.json')['result'])

    def v2_development(ident, model):
        if model in v2d.NEW.values():
            r = read(v2d.STATE / 'fit' / f'{ident}-{model}.json')['result']
            return r, {arm: r[arm]['transfer'] for arm in ('A', 'B')}
        return round_one(ident, model)

    def v2_confirmation(ident, model):
        return (read(v2c.STATE / 'fit' / f'{ident}-{model}.json')['result'],
                read(v2c.STATE / 'transfer' / f'{ident}-{model}.json')['result'])

    yield ('round_one', ROUND_ONE, old['expanded'], old['entropy'], old_paths, round_one)
    yield ('v2_development', ROUND_ONE, v2d.labels(), new_entropy, new_paths, v2_development)
    yield ('v2_confirmation', v2c.STATE, read(v2c.OUT / 'freeze.json')['labels'], new_entropy, new_paths, v2_confirmation)


def score_round(name, state, labels, entropy, paths, load):
    priors = {m: CharacterPrior.load(paths[m]) for m in set(labels.values())}
    answers = read(state / 'evaluator-only/answers.json')
    idents = [i for block in read(state / 'challenge.json')['schedule'] for i in block]
    rows = []
    for ident in idents:
        answer = answers[ident]
        fits = {m: load(ident, m) for m in set(labels.values())}
        row = dict(id=ident, block=answer['block'], language=answer['language'], kind=answer['kind'])
        for arm in ('A', 'B'):
            for scoring in ('per_run', 'one_code'):
                scores = {}
                for label, model in labels.items():
                    fit, held = fits[model]
                    held = held[arm] if scoring == 'per_run' else rescore(held[arm], priors[model])
                    scores[label] = dict(fit_excess=fit[arm]['bits_per_letter'] - entropy[model],
                                         transfer_excess=None if held['bits_per_letter'] is None else held['bits_per_letter'] - entropy[model],
                                         coverage=held['token_coverage'], cap_hit=fit[arm]['cap_hit'])
                decision = dict(full=decide_transfer(scores))
                if answer['kind'] == 'positive':
                    decision['omitted'] = decide_transfer(scores, [l for l in labels if l != answer['language']])
                row[f'{arm}_{scoring}'] = decision
        rows.append(row)
    positives = [r for r in rows if r['kind'] == 'positive']
    negatives = [r for r in rows if r['kind'] != 'positive']
    summary = {}
    for key in ('A_per_run', 'A_one_code', 'B_per_run', 'B_one_code'):
        summary[key] = dict(
            correct=sum(r[key]['full']['accepted'] == r['language'] for r in positives),
            wrong_language=sum(r[key]['full']['accepted'] not in (None, r['language']) for r in positives),
            omitted_accepted=sum(r[key]['omitted']['accepted'] is not None for r in positives),
            negatives_accepted=sum(r[key]['full']['accepted'] is not None for r in negatives),
            lowest_negative_transfer_excess=min(r[key]['full']['transfer']['excess'] for r in negatives
                                                if r[key]['full']['transfer']['excess'] is not None),
            lowest_omitted_transfer_excess=min(r[key]['omitted']['transfer']['excess'] for r in positives
                                               if r[key]['omitted']['transfer']['excess'] is not None))
    return dict(name=name, positives=len(positives), negatives=len(negatives), summary=summary, rows=rows)


def main():
    results = [score_round(*r) for r in rounds()]
    seal(OUT / 'results.json', dict(rounds=results, post_hoc=True, development_only=True, at=time.time()))
    for r in results:
        print(r['name'], f"({r['positives']} positives, {r['negatives']} negatives)")
        for key, s in r['summary'].items():
            print(f"  {key:11s} correct {s['correct']:2d}  wrong {s['wrong_language']}  omitted {s['omitted_accepted']}  "
                  f"negatives {s['negatives_accepted']}  lowest negative {s['lowest_negative_transfer_excess']:.2f}  "
                  f"lowest omitted {s['lowest_omitted_transfer_excess']:.2f}")


if __name__ == '__main__':
    main()

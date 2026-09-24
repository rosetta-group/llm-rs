"""Near-hapax share of Naibbe ciphertext across the encryptor's settings and plaintext languages.

    python -m experiments.naibbe_near_duplicates
See experiments/naibbe-near-duplicates/PROTOCOL.md.
"""
import itertools
import json
from pathlib import Path
import random

from experiments.joint_development import load_vendor
from experiments.language_id import texts as language_texts, take
from experiments.voynich_suspects import near_hapax_rate
from voynich.data import digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/naibbe-near-duplicates'
LETTERS, WINDOW, SEEDS = 40000, 10000, (1, 2)
GRID = dict(respacing=(9, 17, 27), use_78=(False, True), space_removal=(0., .03, .10))
TARGET = .0833


def encrypt(vendor, dense, respacing, use_78, space_removal, seed):
    vendor.RESPACING = respacing; vendor.UNAMBIGUOUS = True
    random.seed(seed)
    tokens = vendor.encrypt_naibbe(dense, vendor.naibbe_tables, vendor.placeholder_to_glyph, use_78=use_78)
    return vendor.respace_line(' '.join(tokens), space_removal).split()


def main():
    vendor = load_vendor(); t = language_texts()
    plains = {lang: ''.join(take(prior, LETTERS)).replace(' ', '')[:LETTERS] for lang, (prior, _, _) in t.items()}
    rows = []
    for lang, (respacing, use_78, space_removal) in itertools.product(plains, itertools.product(*GRID.values())):
        rates, hapax = [], []
        for seed in SEEDS:
            words = encrypt(vendor, plains[lang], respacing, use_78, space_removal, seed)
            for i in (0, WINDOW):
                r = near_hapax_rate(words[i:i + WINDOW]); rates.append(r['near_hapax_share_of_tokens']); hapax.append(r['hapax_share_of_types'])
        rows.append(dict(language=lang, respacing=respacing, deck=78 if use_78 else 56, space_removal=space_removal,
                         near_hapax=sum(rates) / len(rates), near_hapax_min=min(rates), near_hapax_max=max(rates),
                         hapax_share_of_types=sum(hapax) / len(hapax), tokens_per_letter=len(words) / LETTERS))
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}), flush=True)
    best = max(rows, key=lambda r: r['near_hapax'])
    out = dict(target_eva=TARGET, target_v101=.1220, rows=rows, best=best, naibbe_can_match=best['near_hapax'] >= TARGET,
               top10=sorted(rows, key=lambda r: -r['near_hapax'])[:10], protocol_sha256=digest(OUT / 'PROTOCOL.md'), voynich_decoded=False)
    (OUT / 'results.json').write_text(json.dumps(out, indent=2) + '\n')
    print('best', json.dumps(best), 'can match', out['naibbe_can_match'])


if __name__ == '__main__':
    main()

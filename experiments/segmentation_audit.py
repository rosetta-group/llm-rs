"""Separate word-boundary and cipher-piece failures; test one fixed-key reparse."""
import json
from pathlib import Path
import time

from experiments.historical_sources import verify as historical
from experiments.verse_sources import verify as verse
from experiments.segmentation import corpus
from experiments.joint_development import encode, load_vendor, role_units, majority_key
from experiments.joint_development_v3 import segmenter
from experiments.joint_development_v4 import refined, em_settings
from experiments.joint_recovery_v4 import decode, verify, PRIOR_PATH
from voynich.context_reparse import reparse, score
from voynich.data import digest
from voynich.decipher import normalize, edit_distance
from voynich.description_length import CharacterPrior
from voynich.joint_segments_v2 import joint_em
from voynich.lexical_polish import LexicalCost, polish
from voynich.segmentation import boundaries

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/segmentation-audit'
STATE = ROOT / 'artifacts/segmentation-audit'
CODE = ['voynich/context_reparse.py', 'experiments/segmentation_audit.py', 'experiments/segmentation-audit/PROTOCOL.md']


def streams():
    rows, _ = corpus('UD_Italian-ISDT', 'dev')
    return dict(historical=normalize(' '.join(' '.join(r['paragraphs']) for r in historical() if r['split'] == 'dev')),
                modern=normalize(' '.join(' '.join(r['words']) for r in rows)),
                verse=normalize(' '.join(' '.join(r['lines']) for r in verse() if r['split'] == 'dev')))


def reference(text, n=5200):
    dense = text.replace(' ', '')[:n]
    words, length = [], 0
    for word in text.split():
        if length + len(word) > n:
            break
        words.append(word); length += len(word)
    return dense, ' '.join(words), length


def metrics(recovered, plain, dense, word_segmenter, complete_length):
    # Cipher outputs have insertions/deletions; score all 5,200-letter references
    # including their truncated last word consistently. Perfect-letter word audit
    # below separately excludes that fragment.
    reference_words = plain.split()
    tail = dense[complete_length:]
    if tail:
        reference_words.append(tail)
    segmented = word_segmenter.segment(recovered)
    return dict(cer=edit_distance(recovered, dense) / len(dense),
                wer=edit_distance(segmented.split(), reference_words) / len(reference_words), segmented=segmented)


def main():
    OUT.mkdir(parents=True, exist_ok=True); STATE.mkdir(parents=True, exist_ok=True)
    if (OUT / 'results.json').exists():
        raise FileExistsError('Audit already recorded')
    frozen = verify(); s = frozen['settings']; ws = segmenter(); lexical = LexicalCost(ws)
    prior = CharacterPrior.load(PRIOR_PATH); vendor = load_vendor()
    provenance = dict(code_sha256={p: digest(ROOT / p) for p in CODE}, freeze_sha256=digest(ROOT / 'experiments/joint-recovery-v4/freeze.json'),
                      prior_sha256=digest(PRIOR_PATH), word_model_sha256=digest(ROOT / 'artifacts/standard-decipherment/segmenter.json'),
                      beam_width=128, seed=6200, development_only=True, voynich_text_used=False)
    rows = []
    for source, text in streams().items():
        checkpoint = OUT / (source + '.json')
        if checkpoint.exists():
            row = json.loads(checkpoint.read_text())
            if row['provenance'] != provenance:
                raise ValueError('Checkpoint drift')
            rows.append(row); continue
        dense, plain, length = reference(text)
        perfect = ws.segment(dense[:length]); gold, pred = boundaries(plain), boundaries(perfect)
        word_audit = dict(characters=length, words=len(plain.split()), word_errors=edit_distance(perfect.split(), plain.split()),
                          wer=edit_distance(perfect.split(), plain.split()) / len(plain.split()),
                          boundary_f1=2 * len(gold & pred) / max(1, len(gold) + len(pred)))
        print(json.dumps(dict(source=source, perfect_letter_word_audit=word_audit)), flush=True)
        tokens, truth = encode(vendor, dense, 6200)
        base = decode(tokens, prior, lexical, s, 3600)
        mapping = majority_key(role_units(base['segmentation']), base['recovered'])
        if ''.join(mapping[u] for u in role_units(base['segmentation'])) != base['recovered']:
            raise ValueError('Inconsistent fixed key')
        candidate = reparse(tokens, mapping, prior, width=128)
        original_bits = score(base['segmentation'], mapping, prior)
        accepted = candidate['bits'] < original_bits - 1e-8
        chosen = candidate if accepted else base
        print(json.dumps(dict(source=source, candidate_seconds=candidate['seconds'], lower_score=accepted)), flush=True)
        oracle = joint_em(tokens, prior, pieces={p for t in truth for p in t}, **em_settings(s))
        units, ref = refined(tokens, prior, oracle, s)
        pol = polish(units, prior, ref['mapping'], lexical, weight=s['polish_weight'], radius=s['polish_radius'],
                     shortlist=s['polish_shortlist'], sweeps=s['polish_sweeps'], cap=s['polish_cap'])
        def grade(result):
            return dict(**metrics(result['recovered'], plain, dense, ws, length),
                        parse_agreement=sum(tuple(a) == tuple(b) for a, b in zip(result['segmentation'], truth)) / len(truth))
        row = dict(source=source, provenance=provenance, perfect_letters=word_audit, baseline=grade(base),
                   candidate=grade(chosen), oracle=grade(dict(oracle, recovered=pol['recovered'])),
                   baseline_score=original_bits, candidate_score=candidate['bits'], candidate_accepted_by_score=accepted,
                   candidate_seconds=candidate['seconds'], oracle_refined_cer=edit_distance(ref['recovered'], dense) / len(dense),
                   cap_hits=dict(baseline=base['case_cap_hit'] or any(base[k] for k in ('refine_cap_hit',)) or base['polish']['cap_hit']
                                 or any(base['joint'][k] for k in ('first_cap_hit', 'second_cap_hit', 'repaired_cap_hit')),
                                 oracle=oracle['cap_hit'] or ref['cap_hit'] or pol['cap_hit']))
        record = dict(plaintext=plain, dense=dense, ciphertext=tokens, truth=truth, baseline=base,
                      candidate=candidate, oracle=dict(recovered=pol['recovered'], segmentation=oracle['segmentation']),
                      perfect_word_prediction=perfect)
        # No sets or NumPy scalars in these result subsets.
        (OUT / (source + '-predictions.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        checkpoint.write_text(json.dumps(row, indent=2) + '\n'); rows.append(row)
        print(json.dumps({k: row[k] for k in ('source', 'cap_hits')} | {k: {m: row[k][m] for m in ('cer', 'wer', 'parse_agreement')} for k in ('baseline', 'candidate', 'oracle')}), flush=True)
    means = {stage: {metric: sum(r[stage][metric] for r in rows) / len(rows) for metric in ('cer', 'wer')} for stage in ('baseline', 'candidate', 'oracle')}
    selected = (means['baseline']['cer'] - means['candidate']['cer'] >= .01
                and all(r['candidate']['cer'] - r['baseline']['cer'] <= .01 for r in rows)
                and means['candidate']['wer'] <= means['baseline']['wer']
                and not any(any(r['cap_hits'].values()) for r in rows))
    output = dict(provenance=provenance, rows=rows, means=means, selected=selected,
                  fresh_evaluation='required after a committed freeze' if selected else 'not run: candidate failed development selection',
                  final_test_scored=False)
    (OUT / 'results.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(dict(means=means, selected=selected)), flush=True)


if __name__ == '__main__':
    main()

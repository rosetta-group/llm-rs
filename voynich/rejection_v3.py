"""Fixed-key transfer with one length code per passage.

`voynich.rejection.transfer` scores each covered run between unreadable tokens with a separate
`CharacterPrior.bits` call, and each call adds the Elias-gamma code of that run's length. A passage
with 59 gaps then pays about 59 length codes, which adds a median 0.13 bits per letter to its
transfer excess, while the calibration score it is compared with pays one code over 20,000 letters.

Here each run keeps its own context reset (a gap still breaks the character context), but the
passage pays a single length code for all its decoded letters. Everything else is unchanged.
"""
from voynich.description_length import integer_bits
from voynich.rejection import transfer as transfer_per_run


def transfer(tokens, mapping, prior, width=128):
    result = transfer_per_run(tokens, mapping, prior, width=width)
    letters = result['recovered_letters']
    if letters:
        runs = [r for r in result['recovered'].split('?') if r]
        if sum(map(len, runs)) != letters:
            raise ValueError('Run letters do not match the transfer record')
        bits = sum(prior.bits(r) - integer_bits(len(r)) for r in runs) + integer_bits(letters)
        result = dict(result, bits_per_letter=bits / letters, length_code='one per passage')
    return result


def rescore(result, prior):
    """Correct an archived per-run transfer record without decoding again."""
    runs = [r for r in result['recovered'].split('?') if r]
    letters = sum(map(len, runs))
    if not letters:
        return dict(result, bits_per_letter=None, length_code='one per passage')
    bits = sum(prior.bits(r) - integer_bits(len(r)) for r in runs) + integer_bits(letters)
    return dict(result, bits_per_letter=bits / letters, length_code='one per passage')


GROUPS = {'catalan_occitan': ('catalan', 'occitan')}


def group_scores(scores, groups=GROUPS):
    """Merge each group into one candidate: it scores as whichever member fits better on each passage."""
    members = {l for g in groups.values() for l in g}
    out = {l: s for l, s in scores.items() if l not in members}
    for name, group in groups.items():
        present = [l for l in group if l in scores]
        if not present:
            continue
        fit = min(scores[l]['fit_excess'] for l in present)
        held = [(scores[l]['transfer_excess'], l) for l in present if scores[l]['transfer_excess'] is not None]
        best = min(held)[1] if held else present[0]
        out[name] = dict(fit_excess=fit, transfer_excess=min(held)[0] if held else None,
                         coverage=scores[best]['coverage'], cap_hit=any(scores[l]['cap_hit'] for l in present))
    return out


def label(language, groups=GROUPS):
    return next((name for name, group in groups.items() if language in group), language)

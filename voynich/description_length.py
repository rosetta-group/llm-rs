"""A conditional character prior and a lossless two-part candidate code."""
from collections import Counter
import math

import numpy as np

from voynich.decipher import ALPHABET, encode


def integer_bits(n):
    """Elias gamma length for a nonnegative integer, coded as n + 1."""
    if n < 0:
        raise ValueError('Negative length')
    return 2 * int(math.log2(n + 1)) + 1


class CharacterPrior:
    def __init__(self, probabilities, alphabet=ALPHABET):
        self.probabilities = probabilities
        self.logs = [-np.log2(p) for p in probabilities]
        self.alphabet = alphabet
        self.base = len(alphabet)
        self.order = len(probabilities)

    @classmethod
    def fit(cls, texts, order=5, alphabet=ALPHABET, strength=5.):
        base = len(alphabet)
        counts = [np.zeros(base ** n) for n in range(1, order + 1)]
        for text in texts:
            values = encode(text.replace(' ', ''), alphabet)
            ids = values.copy()
            for n in range(1, order + 1):
                if n > len(values):
                    break
                if n > 1:
                    ids = ids[:-1] * base + values[n-1:]
                unique, frequency = np.unique(ids, return_counts=True)
                counts[n-1][unique] += frequency
        probabilities = [(counts[0] + .1) / (counts[0].sum() + .1 * base)]
        for n in range(2, order + 1):
            rows = counts[n-1].reshape(-1, base)
            lower = np.tile(probabilities[-1].reshape(-1, base), (base, 1))
            probabilities.append(((rows + strength * lower) / (rows.sum(axis=1, keepdims=True) + strength)).ravel())
        return cls(probabilities, alphabet)

    def bits(self, text):
        values = encode(text, self.alphabet)
        total = float(integer_bits(len(values)))
        for i in range(min(self.order-1, len(values))):
            index = 0
            for c in values[:i+1]:
                index = index * self.base + c
            total += self.logs[i][index]
        if len(values) >= self.order:
            ids = values[:len(values)-self.order+1].copy()
            for offset in range(1, self.order):
                ids = ids * self.base + values[offset:len(values)-self.order+offset+1]
            total += self.logs[-1][ids].sum()
        return float(total)

    def save(self, path):
        np.savez(path, **{f'p{n}': p for n,p in enumerate(self.probabilities)})

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls([data[f'p{n}'] for n in range(len(data.files))])


def description_length(ciphertext, representation, mapping, prior):
    """Encode normalized ciphertext through its proposed plaintext and explicit key.

    At each plaintext offset, a uniform code selects one compatible cipher symbol.
    This pays for homophones AND ambiguous one/two-letter chunk boundaries.
    The receiver knows the unit count and reconstructs the original normalized cipher.
    """
    if representation not in ('characters', 'units'):
        raise ValueError('Unknown representation')
    tokens = ciphertext.split()
    units = list(''.join(tokens)) if representation == 'characters' else tokens
    if set(mapping) != set(units) or not units:
        raise ValueError('Key must cover exactly the observed inventory')
    if any(not 1 <= len(v) <= 2 or any(c not in prior.alphabet for c in v) for v in mapping.values()):
        raise ValueError('Expected one/two plaintext letters per cipher unit')
    plaintext = ''.join(mapping[u] for u in units)
    inventory_bits = sum(integer_bits(len(s.encode('utf8'))) + 8 * len(s.encode('utf8')) for s in sorted(mapping))
    inventory_bits += integer_bits(len(mapping))
    key_bits = inventory_bits + sum(1 + len(v) * math.ceil(math.log2(prior.base)) for v in mapping.values())
    # Common general mapping code; no free family-specific key prior.
    choice_counts = Counter(mapping.values())
    residual_bits = 0.
    position = 0
    for unit in units:
        choices = sum(choice_counts.get(plaintext[position:position+n], 0)
                      for n in (1, 2) if position+n <= len(plaintext))
        residual_bits += math.log2(choices)
        position += len(mapping[unit])
    assert position == len(plaintext)
    # Character representation must also transmit the removed word boundaries.
    layout_bits = integer_bits(len(tokens))
    if representation == 'characters' and len(tokens) > 1:
        length, boundaries = len(units)-1, len(tokens)-1
        layout_bits += (math.lgamma(length+1)-math.lgamma(boundaries+1)-math.lgamma(length-boundaries+1))/math.log(2)
    plaintext_bits = prior.bits(plaintext)
    overhead_bits = 1 + integer_bits(len(units)) + layout_bits
    return dict(total_bits=plaintext_bits+key_bits+residual_bits+overhead_bits,
                plaintext_bits=plaintext_bits, key_bits=key_bits, residual_bits=residual_bits,
                overhead_bits=overhead_bits, plaintext_characters=len(plaintext), recovered=plaintext)


def infer_mapping(units, plaintext):
    if len(units) != len(plaintext):
        raise ValueError('Single-letter mapping requires equal lengths')
    mapping = {}
    for unit, letter in zip(units, plaintext):
        if unit in mapping and mapping[unit] != letter:
            raise ValueError('Inconsistent deterministic mapping')
        mapping[unit] = letter
    return mapping

"""Polish cost under the v3 segmenter: the frozen LexicalCost with the spelling-model unknown cost."""
import heapq
import math

from voynich.lexical_polish import LexicalCost, LN2


class SpellingLexicalCost(LexicalCost):
    """Minimum segmentation cost in bits under a SpellingSegmenter (elision join does not change cost)."""

    def cost(self, text):
        s = self.s
        if not text:
            return 0.
        states = [{} for _ in range(len(text) + 1)]
        states[0]['<s>'] = 0.
        for start in range(len(text)):
            active = heapq.nsmallest(s.beam if s.bigram else 1, states[start].items(), key=lambda x: (x[1], x[0]))
            states[start] = dict(active)
            unknown = s.unknown_costs(text, start)
            for end, word, probability in s.candidates(text, start):
                for previous, cost in active:
                    if probability is None:
                        extra = unknown[end]
                    else:
                        transition = (s.following.get(previous, {}).get(word, 0) + 20 * probability) / (s.totals.get(previous, 0) + 20)
                        extra = -math.log((1 - s.bigram) * probability + s.bigram * transition)
                    candidate = cost + extra
                    if candidate < states[end].get(word, math.inf):
                        states[end][word] = candidate
        return min(states[-1].values()) / LN2

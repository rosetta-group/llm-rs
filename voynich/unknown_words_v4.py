"""Unknown-word cost from a mixture of two letter-level spelling models (word segmentation v4)."""
import math

from voynich.unknown_words import END, START, SpellingSegmenter


def _tail(history, order):
    return history[len(history) - order + 1:] if order > 1 else ''


class MixtureSegmenter(SpellingSegmenter):
    """SpellingSegmenter whose unknown words are scored by P(w) = sum_i weight_i * P_i(w)."""

    def __init__(self, model, spellings, weights, rate, elision=False, **parameters):
        if len(spellings) != len(weights) or abs(sum(weights) - 1) > 1e-9 or min(weights) <= 0:
            raise ValueError('Expected positive weights summing to one')
        super().__init__(model, spellings[0], rate, elision, **parameters)
        self.spellings, self.log_weights = spellings, [math.log(w) for w in weights]

    def unknown_costs(self, text, start):
        histories = [START * (m.order - 1) for m in self.spellings]
        running = [0.] * len(self.spellings)
        costs = {}
        for end in range(start + 1, min(len(text), start + self.maximum) + 1):
            char = text[end - 1]; logs = []
            for i, m in enumerate(self.spellings):
                running[i] += math.log(m.probability(_tail(histories[i], m.order), char))
                histories[i] += char
                logs.append(self.log_weights[i] + running[i] + math.log(m.probability(_tail(histories[i], m.order), END)))
            peak = max(logs)
            costs[end] = self.rate_cost - (peak + math.log(sum(math.exp(x - peak) for x in logs)))
        return costs

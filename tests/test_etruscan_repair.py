import unittest

import pandas as pd

from etruscan import repair


def entry(word, pos="nom acc", glosses=(), suffix=False, inferred=False):
    return {"Etruscan": word, "POS": pos, "Translations": repr(glosses),
            "Is suffix": suffix, "Is inferred": inferred}


class RepairTest(unittest.TestCase):
    def test_empty_first_row_does_not_hide_real_gloss(self):
        table = pd.DataFrame([entry("lautni", pos=None),
                              entry("lautni", glosses=((True, "freedman"),))])
        self.assertEqual(repair.repaired_labels(table)[0], {"lautni": "KIN"})
        self.assertEqual(repair.repaired_labels(table.iloc[::-1])[0], {"lautni": "KIN"})

    def test_ambiguity_unknown_and_uncertainty_are_not_gold(self):
        table = pd.DataFrame([entry("cesu", glosses=((True, "buried"),)),
                              entry("cesu", glosses=((True, "resided"),)),
                              entry("unknown"), entry("uncertain", glosses=((False, "tomb"),)),
                              entry("apa", glosses=((True, "father"),), suffix=True),
                              entry("madeup", glosses=((True, "father"),), inferred=True),
                              entry("ar[n]th", pos="masc prae"), entry("vel", pos="masc prae")])
        labels, _, audit = repair.repaired_labels(table)
        self.assertEqual(labels, {"vel": "NAME"})
        self.assertIn("cesu", audit["conflicts"])

    def test_whole_family_stays_together(self):
        words = ["clan", "clenar", "larth", "larthal", "ci", "mi", "puia"]
        groups = repair.family_groups(words, {"clan": ["son"], "clenar": ["sons"]}, ["al"])
        self.assertEqual(groups["clan"], groups["clenar"])
        self.assertEqual(groups["larth"], groups["larthal"])
        labels = {w: "OTHER" for w in words}
        folds = repair.folds(labels, groups, 3, "test")
        self.assertEqual(set.union(*folds), set(words))
        for a in range(3):
            for b in range(a):
                self.assertFalse({groups[t] for t in folds[a]} & {groups[t] for t in folds[b]})

    def test_family_label_cannot_leak_through_neighbour_features(self):
        texts = [["larth", "larthal", "clan"]]
        groups = {"larth": "larth", "larthal": "larth", "clan": "clan"}
        first, _ = repair.features(texts, {"larthal": "NAME", "clan": "KIN"}, groups)
        second, _ = repair.features(texts, {"larthal": "LIFE", "clan": "KIN"}, groups)
        self.assertEqual(first["larth"], second["larth"])
        self.assertEqual(first["larth"]["R:UNK"], 1)

    def test_absent_evidence_requires_abstention(self):
        texts = [["seed", "other"], ["apa"]]
        groups = {w: w for tx in texts for w in tx}
        pred = repair.predict(texts, {"seed": "NAME"}, groups)
        self.assertFalse(pred["apa"]["evidence"])
        rows = [dict(pred["apa"], gold="NAME", family=str(i), score=1.0) for i in range(30)]
        self.assertIsNone(repair.choose_threshold(rows))

    def test_no_full_word_suffix_feature(self):
        rows, _ = repair.features([["ci", "clan"]], {}, {"ci": "ci", "clan": "clan"}, endings=True)
        self.assertIn("S1:i", rows["ci"])
        self.assertNotIn("S2:ci", rows["ci"])
        self.assertNotIn("S4:clan", rows["clan"])

    def test_calibration_requires_support_and_quality(self):
        rows = [dict(predicted="NAME", gold="NAME", evidence=True, score=0.9, family=str(i)) for i in range(20)]
        self.assertEqual(repair.choose_threshold(rows), 0.5)
        self.assertIsNone(repair.choose_threshold(rows[:4]))
        wrong = [dict(r, gold="KIN") for r in rows]
        self.assertIsNone(repair.choose_threshold(wrong))


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from voynich.corpora import conllu_sentences, letter_form, voynich_segments, written_words
from experiments.languages import one_edit, profile


class LanguageTests(unittest.TestCase):
    def test_letters_accents_and_internal_boundaries(self):
        self.assertEqual(written_words("L’uomo, déjà-vu; Straße 123!"), ["l'uomo", "déjà-vu", "straße"])
        self.assertEqual(len(letter_form("déjà-vu")), 6)
        self.assertEqual(len(letter_form("straße")), 6)

    def test_conllu_uses_surface_tokens_and_ignores_empty_nodes(self):
        rows = ["# sent_id = a", "1-2\tau\t_\t_\t_\t_\t_\t_\t_\t_",
                "1\tà\t_\tADP\t_\t_\t0\troot\t_\t_", "2\tle\t_\tDET\t_\t_\t1\tdet\t_\t_",
                "2.1\tfake\t_\tX\t_\t_\t_\t_\t0:root\t_",
                "3\tport\t_\tNOUN\t_\t_\t1\tnmod\t_\tSpaceAfter=No",
                "4\t.\t_\tPUNCT\t_\t_\t1\tpunct\t_\t_"]
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "sample.conllu"; p.write_text("\n".join(rows) + "\n\n")
            self.assertEqual(list(conllu_sentences(p))[0]["words"], ["au", "port"])

    def test_voynich_mask_does_not_bridge_and_test_is_excluded(self):
        docs = [dict(page="a", currier="A", section="herbal", split="train", text="aa.?.b,c\ndd.ee"),
                dict(page="b", currier="B", section="herbal", split="test", text="sealed")]
        segments = list(voynich_segments(docs))
        self.assertEqual(segments[0]["words"], ["aa", None, "b", "c"])
        x = profile(segments, glyphs=True, shuffles=3)
        self.assertEqual(x["words"], 5)
        self.assertEqual(x["adjacency_pairs"], 2)
        self.assertEqual(x["excluded_uncertain_forms"], 1)
        self.assertEqual(list(voynich_segments(docs, uncertain="merge"))[0]["words"], ["aa", None, "bc"])

    def test_edit_distance_is_exactly_one(self):
        for a, b in [("ab", "ac"), ("ab", "abc"), ("cab", "ab"), ("abc", "ac")]:
            self.assertTrue(one_edit(a, b))
        for a, b in [("ab", "ab"), ("ab", "cd"), ("ab", "abcd")]:
            self.assertFalse(one_edit(a, b))


if __name__ == "__main__":
    unittest.main()

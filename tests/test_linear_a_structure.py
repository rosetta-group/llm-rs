import unittest

import numpy as np

from linear_a import structure


class StructureTest(unittest.TestCase):
    def test_unknown_sign_retained_but_damage_is_not_a_word_break(self):
        # AB002 (ro), A301 (unknown sound), numeral 1; damaged second line excluded.
        text = '\U00010601\U00010655\U00010107\n\U00010601\U0001076b\U00010655\U00010107'
        rows, _ = structure.sign_tokens({'HT 999': {'unicode_text': text}})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], ('AB02', 'A301'))
        self.assertEqual(rows[0][2], 1)

    def test_faces_group_and_test_words_are_unseen(self):
        self.assertEqual(structure.document_group('HT 1a'), structure.document_group('HT 1b'))
        docs = ['HT '+str(i) for i in range(100)]
        a = next(d for d in docs if not structure.is_test(d))
        b = next(d for d in docs if structure.is_test(d))
        train, test = structure.partition([(a, ('a','b'), 1, 'HT'), (b, ('a','b'), 1, 'HT'),
                                           (b, ('a','c'), 0, 'HT')])
        self.assertEqual([r['word'] for r in test], [('a','c')])
        self.assertFalse({r['word'] for r in train} & {r['word'] for r in test})

    def test_endings_recover_planted_order_signal_that_bag_cannot(self):
        train, test = [], []
        for i in range(100):
            for y in (0,1):
                r = {'word': ('x'+str(i), 'a','b') if y else ('x'+str(i), 'b','a'),
                     'label': y, 'site': 'one'}
                (train if i<70 else test).append(r)
        labels = [r['label'] for r in test]
        self.assertEqual(structure.balanced_accuracy(labels, structure.predict(train,test)), .5)
        self.assertEqual(structure.balanced_accuracy(labels, structure.predict(train,test,True)), 1.)

    def test_shuffle_preserves_bag_and_stratified_class_counts(self):
        rows = [{'word': ('x', str(i), 'z'), 'label': i%2, 'site':'s'} for i in range(10)]
        rng = np.random.default_rng(3)
        shuffled = structure.shuffled_words(rows,rng)
        self.assertEqual([structure.features(r['word']) for r in rows],
                         [structure.features(r['word']) for r in shuffled])
        self.assertEqual(sum(r['label'] for r in rows),
                         sum(r['label'] for r in structure.permuted_training(rows,rng)))


if __name__ == '__main__':
    unittest.main()

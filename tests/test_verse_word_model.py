import copy
import unittest
from voynich.segmentation import fit
from voynich.verse_word_model import augment
from experiments.word_segmentation_v2 import grade,choose

class VerseWordTests(unittest.TestCase):
    def test_train_only_weighted_merge_preserves_base(self):
        base=fit(['la casa'],[]);before=copy.deepcopy(base)
        rows=[dict(id='train1',split='train',lines=['la fronde la']),dict(id='dev1',split='dev',lines=['segreto'])]
        model,ids=augment(base,rows,4)
        self.assertEqual(base,before);self.assertEqual(ids,['train1'])
        self.assertEqual(model['counts']['la'],9);self.assertEqual(model['following']['la']['fronde'],4)
        self.assertNotIn('segreto',model['lexicon']);self.assertIn('fronde',model['lexicon'])

    def test_duplicates_and_invalid_weights_fail(self):
        r=dict(id='a',split='train',lines=['parola'])
        with self.assertRaises(ValueError):augment(fit([],[]),[r,r],1)
        with self.assertRaises(ValueError):augment(fit([],[]),[r],0)

    def test_grade_rejects_changed_letters(self):
        with self.assertRaises(ValueError):grade('la cosa','la casa')
        self.assertEqual(grade('l acasa','la casa')['character_error'],0)
        self.assertEqual(grade('la casa','la casa')['wer'],0)

    def test_selection_requires_gain_and_modern_guard(self):
        row=lambda w,h,v,m:dict(weight=w,grades={s:dict(wer=x) for s,x in zip(('historical','verse','modern'),(h,v,m))})
        base=row(0,.15,.28,.07)
        self.assertIsNone(choose([base,row(1,.14,.26,.07)]))
        self.assertIsNone(choose([base,row(1,.14,.15,.09)]))
        self.assertEqual(choose([base,row(4,.14,.15,.07),row(1,.14,.15,.07)]),1)

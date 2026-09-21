import unittest
import numpy as np
from experiments.association import endpoint, parse, eligible, prediction_operator, permute_within

class AssociationTests(unittest.TestCase):
    def test_unknown_is_not_absent(self):
        self.assertIsNone(endpoint('green leaves'))
        self.assertIsNone(endpoint('light root?'))
        self.assertIsNone(endpoint('light roots and dark roots'))
        self.assertEqual(endpoint('dark coloured roots'),1)
        self.assertEqual(endpoint('light roots'),0)

    def test_dedup_and_sealed_folio(self):
        rows=parse('1|pharma|f88r|A|1|C|abc|-|P|plant|light roots\n2|pharma|f88r|A|1|V|xyz|-|P|plant|light roots')
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['label'],'xyz')
        meta={'f88r':dict(folio='88',hand='1')}
        self.assertEqual(eligible(rows,meta,{'88':'test'}),[])
        self.assertEqual(len(eligible(rows,meta,{'88':'train'})),1)

    def test_heldout_targets_cannot_affect_prediction(self):
        x=np.random.default_rng(1).normal(size=(12,5)); groups=np.repeat([1,2,3],4)
        a=prediction_operator(x,groups)
        for group in (1,2,3):
            ix=np.flatnonzero(groups==group)
            np.testing.assert_array_equal(a[np.ix_(ix,ix)],0)
        y=np.arange(12)/12; train=groups!=1;test=~train
        mean=x[train].mean(0);scale=x[train].std(0)
        xt=(x[train]-mean)/scale
        beta=np.linalg.solve(xt.T@xt+10*np.eye(5),xt.T@(y[train]-y[train].mean()))
        np.testing.assert_allclose((a@y)[test],((x[test]-mean)/scale)@beta+y[train].mean())

    def test_permutation_preserves_page_counts(self):
        y=np.array([0,0,1,1,0,1]);pages=np.array(['a']*3+['b']*3)
        result=permute_within(y,pages,np.random.default_rng(42))
        for page in set(pages):
            self.assertEqual(sorted(y[pages==page]),sorted(result[pages==page]))

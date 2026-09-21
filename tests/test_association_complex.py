import unittest
import numpy as np
from voynich.association_complex import BASE,describe,operators,ranks,holm,target_folds,profile_scores,ridge_operator,select_profiles
from experiments.association import permute_within

class ComplexAssociationTests(unittest.TestCase):
    def test_uncertain_and_editorial_clauses(self):
        p=describe('light roots, dark leaves?, label has three letters, dark flower')
        names={n for n,v in zip(BASE,p) if v}
        self.assertEqual(names,{'root','root_light','flower','flower_dark'})
        self.assertGreater(p[len(BASE):].sum(),0)
        self.assertEqual(describe('uncertain plant?').sum(),0)

    def test_heldout_targets_and_xor(self):
        x=np.tile(np.array([[-1,-1],[-1,1],[1,-1],[1,1]])/np.sqrt(2),(6,1))
        y=(x[:,0]*x[:,1]>0).astype(float);groups=np.repeat(np.arange(6),4)
        maps=operators(np.zeros((24,1)),x,groups)
        for a in maps.values():
            for g in set(groups):
                ix=np.flatnonzero(groups==g);np.testing.assert_array_equal(a[np.ix_(ix,ix)],0)
        self.assertTrue(np.all((maps['interactions']@y>=.5)==y))
        self.assertTrue(np.all((maps['nonlinear']@y>=.5)==y))
        np.testing.assert_allclose(maps['additive']@y,.5,atol=1e-10)

    def test_tied_matching_and_holm(self):
        np.testing.assert_allclose(ranks(np.ones((4,2)),np.ones((4,2))),.5)
        np.testing.assert_allclose(ranks(np.eye(4),np.eye(4)),1.)
        self.assertEqual(holm({'a':.01,'b':.04,'c':.03}),{'a':.03,'c':.06,'b':.06})

    def test_joint_permutations_preserve_fold_support(self):
        y=np.tile(np.eye(3),(8,1));pages=np.repeat(np.arange(6),4);groups=pages//2
        yp=permute_within(y,pages,np.random.default_rng(1))
        for g in set(groups):np.testing.assert_array_equal(y[groups!=g].sum(0),yp[groups!=g].sum(0))
        folds=target_folds(y,groups,pages)
        scores=profile_scores(y,y,folds)
        np.testing.assert_allclose(scores['mse'],0)

    def test_kernel_operator_matches_explicit_primal_ridge(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(14,4));y=rng.normal(size=(14,3))
        train=np.arange(9);test=np.arange(9,14)
        design=np.column_stack([np.ones(14),x]);penalty=np.diag([0.,1.,1.,1.,1.])
        beta=np.linalg.solve(design[train].T@design[train]+penalty,design[train].T@y[train])
        predicted=ridge_operator(x@x.T,train,test)@y[train]
        np.testing.assert_allclose(predicted,design[test]@beta,atol=1e-12)

    def test_profile_split_and_mixed_colors(self):
        row=dict(page='f101v1',group='A',index=1,transcriber='V',label='abc',
                 object='plant',section='pharma',description='light and dark roots')
        metadata={'f101v':dict(folio='101',hand='1')}
        self.assertEqual(select_profiles([row],metadata,{'101':'test'}),[])
        selected=select_profiles([row],metadata,{'101':'train'})
        self.assertEqual(len(selected),1)
        for feature in ('root_light','root_dark'):
            self.assertEqual(selected[0]['profile'][BASE.index(feature)],1)
        self.assertEqual(select_profiles([dict(row,object='plant?')],metadata,{'101':'train'}),[])

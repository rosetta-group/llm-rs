import unittest
import numpy as np
from voynich.image_domains import aggregate,forms,hashed_counts,score,shuffle,movable,operators,predictions

class ImageDomainTests(unittest.TestCase):
    def line(self,folio,page,section,raw='abc.def'):
        return dict(folio=folio,page=page,section=section,quire='N',hand='2',locator='@',kind='P',raw=raw)

    def test_split_before_normalization_and_rosettes(self):
        lines=[self.line('85','f85r2','cosmological'),self.line('Ros','fRos','cosmological'),
               self.line('86','f86v3','cosmological'),self.line('4','f4r','herbal','@999;')]
        rows,audit=aggregate(dict(lines=lines),{'85':'train','86':'train','Ros':'train','4':'test'})
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['folio'],'85-86');self.assertEqual(len(rows[0]['pages']),3)
        self.assertEqual(audit['training_pages'],3)

    def test_annotations_and_word_boundaries(self):
        self.assertEqual(forms('<%>abc.def<$>')[0],['abc','def'])
        self.assertEqual(forms('abc.a?c.*')[0],['abc'])
        a=hashed_counts([{'tokens':['a','b']}],'characters')
        b=hashed_counts([{'tokens':['ab']}],'characters')
        self.assertEqual(a.sum(),2);self.assertEqual(b.sum(),3)

    def test_macro_recall_and_strata(self):
        y=np.array([0,0,0,1]);pred=np.zeros(4,dtype=int)
        self.assertEqual(score(y,pred,2)['macro_recall'],.5)
        with self.assertRaises(ValueError):score(np.array([0]),np.array([0]),2)
        strata=['a','a','b','b'];yp=shuffle(y,strata,np.random.default_rng(3))
        self.assertEqual(sorted(yp[:2]),sorted(y[:2]));self.assertEqual(sorted(yp[2:]),sorted(y[2:]))
        self.assertEqual(movable(y,strata)['movable_folios'],2)

    def test_synthetic_domain_and_fold_isolation(self):
        rows=[]
        for i in range(18):
            domain=i%3;tokens=[['aaaa','abab'],['mmmm','mnmn'],['xxxx','xyxy']][domain]*10
            rows.append(dict(tokens=tokens,loci=10,pages=['p'],unknown=0,hands={'1':10},kinds={'P':10},position=i,quire=str(i//3)))
        groups=np.arange(18);maps=operators(rows,groups);y=np.arange(18)%3
        for a in maps.values():np.testing.assert_array_equal(np.diag(a),0)
        self.assertEqual(score(y,predictions(maps['words'],y,3),3)['macro_recall'],1)
        self.assertEqual(score(y,predictions(maps['characters'],y,3),3)['macro_recall'],1)

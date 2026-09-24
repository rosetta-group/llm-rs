from copy import deepcopy
from fractions import Fraction
import unittest
from linear_a.account_benchmark import quantity, sum_items, inspect


def row(dim, commodity, **parts):
    return {'dimension':dim,'commodity':commodity,'components':[
        {'unit':u,'n':n,'certainty':'certain'} for u,n in parts.items()]}


class AccountBenchmarkTests(unittest.TestCase):
    def test_weight_carry_with_target_hidden(self):
        a={'items':[row('weight','AES',M=1,N=2) for _ in range(8)],'complete':True}
        assert sum_items(a)==Fraction(12,30)  # No target or operation label exists.


    def test_volume_systems_differ(self):
        assert quantity(row('liquid','OLE',V=18))==1
        assert quantity(row('dry','HORD',V=18))==Fraction(3,10)
        assert quantity(row('liquid','OLE',S=2,V=6))==1


    def test_missing_and_restored_are_not_zero_or_certain(self):
        r=row('count','VIR',ONE=None)
        assert quantity(r) is None
        r=row('count','VIR',ONE=1)
        r['components'][0]['certainty']='restored'
        assert quantity(r) is None
        assert quantity(r,False)==1
        assert quantity({'dimension':'count','components':[]}) is None


    def test_incomplete_block_only_has_visible_sum(self):
        a={'items':[row('count','VIR',ONE=3)],'complete':False}
        assert sum_items(a) is None
        assert sum_items(a,False)==3


    def test_dimensions_and_commodities_cannot_be_added(self):
        a={'items':[row('count','VESSEL',ONE=38)],'target':row('liquid','OLE',BASE=18),
           'complete':True,'id':'test','object':'test'}
        assert inspect(a)['strict']['status']=='incompatible'
        a['items'].append(row('count','VIR',ONE=1))
        assert sum_items(a) is None


    def test_target_mutation_does_not_change_prediction(self):
        a={'items':[row('weight','AES',M=5)],'target':row('weight','AES',M=5),'complete':True}
        b=deepcopy(a);b['target']=row('weight','AES',L=99)
        assert sum_items(a)==sum_items(b)


    def test_invalid_unit_and_negative_quantity_fail(self):
        with self.assertRaises(KeyError): quantity(row('liquid','OLE',T=1))
        with self.assertRaises(ValueError): quantity(row('count','VIR',ONE=-1))

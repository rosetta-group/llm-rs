import unittest
from linear_a.kinship_inventory import counted_words, marker_hits

class KinshipInventoryTests(unittest.TestCase):
    def test_word_order_and_repeated_slots_survive(self):
        table={'AB01':('da',0),'AB02':('ro',0)}
        self.assertEqual(counted_words('𐘁𐄁𐘀𐄁𐘁𐄈',table),
                         {'words':['AB02','AB01','AB02'],'count':2})

    def test_damage_fraction_internal_quantity_and_commodity_rejected(self):
        table={'AB01':('da',0),'AB02':('ro',1)}
        for line in ['𐘀[𐄈','𐘀𐝀𐄈','𐘀𐄇𐘀𐄇','𐘀𐄁𐘁𐄈']:
            self.assertIsNone(counted_words(line,table))

    def test_literal_marker_counts_do_not_merge_suffix_or_damage(self):
        got=marker_hits([('1',{'content':'.1 i-jo i-jo i-jo-qe a-i-jo i-jo[ tu-ka-te-qe'})])
        self.assertEqual(got['i-jo']['occurrences'],2)
        self.assertEqual(got['i-jo']['objects'],1)
        self.assertEqual(got['i-jo-qe']['occurrences'],1)

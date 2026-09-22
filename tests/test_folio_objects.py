import copy
import json
from pathlib import Path
import unittest
from voynich.folio_objects import describe, validate, DOMAINS, PATTERNS
from experiments.folio_agreement import binary_agreement, compare

ROOT=Path(__file__).resolve().parents[1]

class ObjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows={r['folio_id']:r for r in json.loads((ROOT/'data/folios/object-pilot/observations.json').read_text())['panels']}

    def test_linked_occupied_basins_require_two_distinct_basins(self):
        r=copy.deepcopy(self.rows['f81r'])
        self.assertIn('figures_in_linked_basins',describe(r)['relation_patterns'])
        r['relations']=[x for x in r['relations'] if x['object']!='lower_basin']
        self.assertNotIn('linked_basins',describe(r)['relation_patterns'])

    def test_stars_in_text_margin_do_not_imply_celestial(self):
        self.assertNotIn('celestial_like',describe(self.rows['f103r'])['domains'])
        self.assertIn('celestial_like',describe(self.rows['f67r1'])['domains'])
        self.assertNotIn('botanical',describe(self.rows['f1r'])['domains'])

    def test_tentative_relations_remain_tentative(self):
        self.assertIn('linked_circular_structures',describe(self.rows['fRos'])['tentative_patterns'])
        self.assertIn('human_figures',describe(self.rows['f57v'])['tentative_domains'])

    def test_all_rows_valid_and_bad_boxes_rejected(self):
        for r in self.rows.values(): validate(r)
        r=copy.deepcopy(self.rows['f81r']);r['objects'][0]['bbox']=[.5,0,.2,1]
        with self.assertRaises(ValueError): validate(r)

    def test_agreement_missingness_and_constant_labels(self):
        got=binary_agreement([True,True,False,None],[True,False,False,True])
        self.assertAlmostEqual(got['agreement'],2/3);self.assertAlmostEqual(got['kappa'],.4)
        self.assertEqual(got['unassessable'],1)
        self.assertIsNone(binary_agreement([False],[False])['kappa'])
        self.assertIsNone(binary_agreement([None],[True])['agreement'])

    def test_unfilled_forms_cannot_be_used_as_agreement(self):
        with self.assertRaises(ValueError): compare({'status':'pending'},{'status':'pending'})

    def test_complete_forms_align_by_panel_id(self):
        row=lambda i:dict(panel_id=i,domains={d:False for d in DOMAINS},patterns={p:False for p in PATTERNS})
        a=dict(status='complete',annotator_type='human',annotator_id='A',independent=True,text_masked=True,mask_reviewed=True,
               mask_manifest_sha256='a'*64,analysis_id='test',panels=[row('1'),row('2')])
        a['panels'][0]['domains']['botanical']=True
        b=copy.deepcopy(a);b['annotator_id']='B';b['panels'].reverse()
        self.assertEqual(compare(a,b)['scores']['domains']['botanical']['kappa'],1)

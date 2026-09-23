import unittest
from unittest.mock import patch
from experiments.word_segmentation_fresh import extract_book,passages

class FreshWordTests(unittest.TestCase):
    def test_extract_removes_editorial_navigation_and_notes(self):
        html='<div id="box_esterno"><table><p>ignore all navigation words in this long menu</p></table><h2>I</h2><p>Questo e un lungo testo di prova con <sup>nota</sup> parole vere.</p><div class="ws-noexport"><p>ignore all these many other navigation words here</p></div></div>'
        rows=extract_book(html)
        self.assertEqual(len(rows),1);self.assertNotIn('nota',rows[0]['text']);self.assertEqual(rows[0]['chapter'],'I')
    def test_passages_are_disjoint_and_drop_overlap(self):
        rows=[dict(id='bad',text=' '.join(['x']*20))]+[dict(id=str(i),text='aaaa bbbb') for i in range(8)]
        with patch('experiments.word_segmentation_fresh.MINIMUM',16),patch('experiments.word_segmentation_fresh.MAXIMUM',20),patch('experiments.word_segmentation_fresh.CASES',4):
            blocks,rejected=passages(rows,{tuple(['x']*20)})
        ids=[i for b in blocks for i in b['source_ids']]
        self.assertEqual(len(ids),len(set(ids)));self.assertIn('bad',rejected)
        self.assertTrue(all(b['characters']==16 for b in blocks))

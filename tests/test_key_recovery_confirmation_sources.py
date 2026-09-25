"""Normalization, overlap filter and passage cut of the confirmation sources, on tiny synthetic inputs."""
import json
import unittest

from experiments import key_recovery_confirmation_sources as k
from experiments.language_coverage_sources import chunks, normalize_source
from experiments.rejection_transfer_v2 import grams
from voynich.decipher import ALPHABET


def doc(text, ident='d'):
    return list(chunks(dict(id=ident, group=ident, text=text), words_per_chunk=10))


def words(prefix, n):
    # Three letters per word, drawn from the cipher alphabet so normalization keeps them unchanged.
    return ' '.join(prefix + ALPHABET[i // len(ALPHABET)] + ALPHABET[i % len(ALPHABET)] for i in range(n))


class NormalizationTests(unittest.TestCase):
    def test_czech_style_normalization(self):
        self.assertEqual(normalize_source('Kněz Wáclav jest ſvatý'), 'cnez uuaclav iest svaty')

    def test_czech_window_allows_1520_and_strips_header(self):
        meta, body = k.czech_document('x', '# title: T\n# originDate: 1500--1520\nPraktika [...] testamentu')
        self.assertEqual(meta['title'], 'T')
        self.assertEqual(normalize_source(body), 'practica testamentu')
        for date in ['1521', '1290', '']:
            with self.subTest(date=date), self.assertRaises(ValueError):
                k.czech_document('x', f'# title: T\n# originDate: {date}\nText')

    def test_occitan_hyphenation_and_lacunae(self):
        self.assertEqual(normalize_source(k.occitan_clean('benau-\nrada [...] sancta')), 'benaurada sancta')

    def test_openmedfr_strips_apparatus(self):
        raw = ('﻿#META# title\nPageV01P001\nEXPLICIT LI ROMANS\n12Li rois .xvj. chevaliers (fol. 3 a)\n'
               'mande [...] *tost§ a sa cort-\nmout grant\n')
        self.assertEqual(normalize_source(k.openmedfr_clean(raw)), 'li rois chevaliers mande tost a sa cortmout grant')

    def test_geste_strips_glued_line_numbers_and_numerals(self):
        self.assertEqual(normalize_source(k.geste_clean('10Qui ot . VII . fius\n\n11Et Huon')), 'qui ot fius et huon')

    def test_dubliners_starts_at_first_story_and_drops_titles(self):
        pg = ('header\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nDUBLINERS\nContents\n'
              '\nTHE SISTERS\nThere was no hope for him this time.\nAN ENCOUNTER\nIt was Joe Dillon.\n'
              '*** END OF THE PROJECT GUTENBERG EBOOK X ***\nfooter')
        self.assertEqual(normalize_source(k.dubliners(pg)), 'there uuas no hope for him this time it uuas ioe dillon')

    def test_wikisource_keeps_prose_paragraphs_only(self):
        html = ('<div class="mw-parser-output"><h2>Caput I</h2><p>TITULUS TOTUS MAIUSCULIS</p>'
                '<p>Karolus <sup>1</sup>rex [7b] fuit (2) in Francia et regnavit annis multis cum gloria magna.</p>'
                '<p>Brevis sententia.</p><div class="poem"><p>versus unus duo tres quattuor quinque sex septem</p></div>'
                '<p><span typeof="mw:File"><img alt="E"/></span>t dixit ad eos verba multa et bona in illo tempore.</p></div>')
        raw = json.dumps(dict(parse=dict(text=html, revid=7, title='T'))).encode()
        revid, title, paragraphs = k.ws_paragraphs(raw, dict(drop=None, sub=None))
        self.assertEqual((revid, title), (7, 'T'))
        self.assertEqual([normalize_source(p) for p in paragraphs],
                         ['carolus rex fuit in francia et regnavit annis multis cum gloria magna',
                          'et dixit ad eos verba multa et bona in illo tempore'])

    def test_latin_share(self):
        self.assertEqual(k.marker_rate('et daz deus got'), 0.5)
        self.assertEqual(k.latin_share('et daz deus got', latin_rate=0.25), 1.0)
        self.assertEqual(k.latin_share('in die ab per daz got uns sal', latin_rate=0.25), 0.0)
        self.assertAlmostEqual(k.latin_share('et daz got uns sal der die wir', latin_rate=0.25), 0.5)


class FilterTests(unittest.TestCase):
    def test_chunk_with_reference_shingle_is_removed_and_tail_is_checked(self):
        rows = doc(words('a', 30))
        blocked_text = ' '.join(rows[1]['text'].split()[2:10])
        reference = grams(blocked_text)
        kept, removed = k.filter_rows(rows, lambda g: 'reference' if g & reference else None)
        self.assertEqual([r['id'] for r in kept], ['d:0', 'd:20'])
        self.assertEqual(removed, [dict(id='d:10', letters=len(rows[1]['text'].replace(' ', '')), reason='reference')])
        # A shingle spanning the join of two kept chunks is caught through the 7-word tail.
        cross = grams(' '.join((rows[0]['text'] + ' ' + rows[1]['text']).split()[5:13]))
        kept, removed = k.filter_rows(rows, lambda g: 'x' if g & cross else None)
        self.assertEqual([r['id'] for r in removed], ['d:10'])

    def test_tail_skips_removed_chunks(self):
        rows = doc(words('b', 30))
        artificial = grams(' '.join(rows[0]['text'].split()[-4:] + rows[2]['text'].split()[:4]))
        middle = grams(rows[1]['text'])
        kept, removed = k.filter_rows(rows, lambda g: 'r' if g & (middle | artificial) else None)
        self.assertEqual([r['id'] for r in removed], ['d:10', 'd:20'])


class CutTests(unittest.TestCase):
    def test_cut_takes_exact_letters_and_whole_rows(self):
        rows = doc(words('c', 25))          # 3 letters per word, 10-word chunks: 30, 30, 15 letters
        p = k.cut(rows, budget=40)
        self.assertEqual(set(p), {'document', 'plaintext', 'rows'})
        self.assertEqual(p['document'], 'd')
        self.assertEqual(len(p['plaintext']), 40)
        self.assertEqual(p['plaintext'], (rows[0]['text'] + rows[1]['text']).replace(' ', '')[:40])
        self.assertEqual(p['rows'], rows[:2])
        with self.assertRaises(ValueError):
            k.cut(rows, budget=76)

    def test_after_gap(self):
        rows = doc(words('e', 100))          # ten 30-letter chunks
        later, skipped = k.after_gap(rows, 'd:10', gap=60)
        self.assertEqual(skipped, 60)
        self.assertEqual(later[0]['id'], 'd:40')
        later, skipped = k.after_gap(rows, 'd:0', gap=61)
        self.assertEqual((later[0]['id'], skipped), ('d:40', 90))
        with self.assertRaises(ValueError):
            k.after_gap(rows, 'd:80', gap=31)

    def test_catalogue_is_fixed(self):
        counts = {l: sum(w['language'] == l for w in k.WORKS) for l in k.LANGUAGES}
        self.assertEqual(counts, dict(latin=6, italian=6, catalan=6, old_french=6, english=6, german=6, czech=6, occitan=4))
        self.assertEqual(len({w['id'] for w in k.WORKS}), len(k.WORKS))
        self.assertTrue(set(k.SECOND_PASSAGES) <= {w['id'] for w in k.WORKS if w['language'] == 'occitan'})

    def test_dumps_matches_seal_format(self):
        self.assertEqual(k.dumps({'b': 1, 'a': [1]}), '{\n  "a": [\n    1\n  ],\n  "b": 1\n}\n')


if __name__ == '__main__':
    unittest.main()

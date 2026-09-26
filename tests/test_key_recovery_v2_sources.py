"""Role assignment, water-filling, grouping and filters of the second confirmation's sources, on synthetic data."""
import hashlib
import json
import unittest
from collections import Counter

from experiments import key_recovery_confirmation_sources as kr
from experiments import key_recovery_v2_sources as v
from experiments.language_coverage_sources import chunks, exact_take, normalize_source
from experiments.rejection_transfer_v2 import grams
from voynich.decipher import ALPHABET


def words(prefix, n):
    # Three letters per word, drawn from the cipher alphabet so normalization keeps them unchanged.
    return ' '.join(prefix + ALPHABET[i // len(ALPHABET)] + ALPHABET[i % len(ALPHABET)] for i in range(n))


def doc(text, ident='d'):
    return list(chunks(dict(id=ident, group=ident, text=text), words_per_chunk=10))


def by_hash(ids):
    return sorted(ids, key=lambda i: hashlib.sha256(i.encode()).hexdigest())


class RoleTests(unittest.TestCase):
    ids = [f'g{i}' for i in range(12)]

    def test_hash_order_test_calibration_train(self):
        roles = v.assign_roles([dict(id=i, clean=50_000) for i in self.ids], 6, 2)
        order = by_hash(self.ids)
        self.assertEqual([roles[i]['role'] for i in order], ['test'] * 6 + ['calibration'] * 2 + ['train'] * 4)
        self.assertEqual([roles[i]['order'] for i in order], list(range(1, 13)))
        self.assertEqual(roles[order[0]]['sha256'], hashlib.sha256(order[0].encode()).hexdigest())

    def test_train_only_and_small_groups_are_skipped_into_train(self):
        order = by_hash(self.ids)
        groups = [dict(id=i, clean=50_000) for i in self.ids]
        groups[self.ids.index(order[0])]['train_only'] = 'OCR'
        groups[self.ids.index(order[2])]['clean'] = 9_999
        roles = v.assign_roles(groups, 6, 2, min_letters=10_000)
        self.assertEqual(roles[order[0]], dict(role='train', order=1, sha256=roles[order[0]]['sha256'],
                                               not_eligible='OCR'))
        self.assertEqual(roles[order[2]]['role'], 'train')
        self.assertEqual(roles[order[2]]['not_eligible'], 'fewer than 10000 clean letters')
        self.assertEqual([roles[i]['role'] for i in order],
                         ['train', 'test', 'train'] + ['test'] * 5 + ['calibration'] * 2 + ['train'] * 2)

    def test_unchanged_prior_leaves_rest_unused_and_tiers_come_first(self):
        groups = [dict(id=i, clean=1, tier=0 if i in ('g10', 'g11') else 1) for i in self.ids]
        roles = v.assign_roles(groups, 6)
        firsts = sorted(('g10', 'g11'), key=lambda i: roles[i]['order'])
        self.assertEqual([roles[i]['order'] for i in firsts], [1, 2])
        self.assertEqual(Counter(r['role'] for r in roles.values()), Counter(test=6, unused=6))

    def test_too_few_eligible_groups_stop(self):
        with self.assertRaises(ValueError):
            v.assign_roles([dict(id=i, clean=1) for i in self.ids[:7]], 6, 2)
        with self.assertRaises(ValueError):
            v.assign_roles([dict(id=i, clean=1, train_only='OCR' if n < 7 else None)
                            for n, i in enumerate(self.ids)], 6, 2)


class WaterFillTests(unittest.TestCase):
    def test_cap_fills_budget_exactly(self):
        take, cap = v.water_fill(dict(a=10, b=50, c=100, d=1000), 200)
        self.assertEqual((take, cap), (dict(a=10, b=50, c=70, d=70), 70))

    def test_remainder_goes_to_first_capped_groups(self):
        take, cap = v.water_fill(dict(x=3, y=3, z=3), 8)
        self.assertEqual((take, cap), (dict(x=3, y=3, z=2), 2))
        take, _ = v.water_fill(dict(x=1, y=5, z=5), 8)
        self.assertEqual(take, dict(x=1, y=4, z=3))

    def test_small_pool_raises(self):
        with self.assertRaises(ValueError):
            v.water_fill(dict(a=5, b=5), 11)

    def test_group_takes_first_letters(self):
        rows = doc(words('a', 30))
        taken = exact_take(rows, 35)
        self.assertEqual(''.join(r['text'] for r in taken), ''.join(r['text'] for r in rows).replace(' ', '')[:35])
        self.assertEqual([r['id'] for r in taken], ['d:0', 'd:10'])


class GroupingTests(unittest.TestCase):
    def test_rem_groups_and_families(self):
        self.assertEqual(v.rem_group('M205A'), 'rem:M205')
        self.assertEqual(v.rem_group('M121Y'), 'rem-family:kaiserchronik')
        self.assertEqual(v.rem_group('M213'), 'rem-family:kaiserchronik')
        self.assertEqual(v.rem_group('M087'), 'rem-family:genesis')
        self.assertEqual(v.rem_group('M999', {}), 'rem:M999')
        members = [m for ms, _ in v.REM_FAMILIES.values() for m in ms]
        self.assertEqual(len(members), len(set(members)))

    def test_pick_takes_largest_work(self):
        works = [dict(id='b', clean=5), dict(id='a', clean=5), dict(id='c', clean=4)]
        self.assertEqual(v.pick(works)['id'], 'a')

    def test_group_entry_carries_train_only(self):
        entry = v.group_entry('g', [dict(clean=3, train_only='OCR'), dict(clean=4)], tier=1)
        self.assertEqual(entry, dict(id='g', clean=7, train_only='OCR', tier=1))


class FilterTests(unittest.TestCase):
    def test_test_blocker_reasons(self):
        a, b, c = doc(words('a', 30), 'A'), doc(words('b', 30), 'B'), doc(words('c', 30), 'C')
        shared = ' '.join(a[1]['text'].split()[:8])
        b[2] = dict(b[2], text=shared + ' ' + ' '.join(b[2]['text'].split()[8:]))     # B copies one shingle of A
        own = {'A': grams(' '.join(r['text'] for r in a)), 'B': grams(' '.join(r['text'] for r in b))}
        counts = Counter(g for s in own.values() for g in s)
        R = grams(' '.join(a[0]['text'].split()[:8]))
        role = grams(' '.join(c[0]['text'].split()[:8]))
        role |= grams(' '.join(a[2]['text'].split()[1:9]))
        kept, removed = kr.filter_rows(a, v.test_blocker(R, role, own, counts, own['A']))
        self.assertEqual([(r['id'], r['reason']) for r in removed],
                         [('A:0', 'reference'), ('A:10', 'other test work'), ('A:20', 'v2 train/calibration')])
        self.assertEqual(kept, [])
        kept, removed = kr.filter_rows(b, v.test_blocker(set(), set(), own, counts, own['B']))
        self.assertEqual([r['id'] for r in removed], ['B:20'])

    def test_catalan_latin_rate(self):
        self.assertEqual(v.ca_latin_rate('et dei gratia rex'), 1.0)
        self.assertEqual(v.ca_latin_rate('lo rey dix a sos cavallers'), 0.0)


class ExtractionTests(unittest.TestCase):
    def test_ws_blocks_prose_drops_migne_lines_and_roman_numerals(self):
        html = ('<div id="mw-content-text"><div class="mw-parser-output"><p>J. P. Migne cc_id 12 editio</p>'
                '<p>Anno MCCXLVII rex [3] venit (2) in urbem et cum eo multi milites .xiiii. et episcopi.</p>'
                '<p>Brevis.</p></div></div>')
        self.assertEqual([normalize_source(p) for p in v.ws_blocks(html.encode(), dict(verse=False))],
                         ['anno rex venit in urbem et cum eo multi milites et episcopi'])

    def test_ws_blocks_verse_lines(self):
        html = ('<div class="mw-parser-output"><div class="poem"><p>12 Karles li reis<br/>'
                'nostre emperere magnes<br/>x</p></div></div>')
        self.assertEqual([normalize_source(l) for l in v.ws_blocks(html.encode(), dict(verse=True))],
                         ['carles li reis', 'nostre emperere magnes'])

    def test_ia_clean_region_and_apparatus(self):
        text = '\n'.join(['intro der text ist', 'INCIPIT', '1 So es lo romans de Daurel e de Beto', '— B. var.',
                          'DAUREL E DE BETO', '12', '2 Senhors, auiatz que Dieus vos benazia 3', 'VOCABULAIRE'])
        w = dict(language='occitan', kind='verse', region=(r'^INCIPIT', r'^VOCABULAIRE'), header=r'DAUREL\s+E\s+DE\s+BETO')
        lines, span = v.ia_clean(text, w)
        self.assertEqual(span, (1, 7))
        self.assertEqual(lines, ['So es lo romans de Daurel e de Beto', 'Senhors, auiatz que Dieus vos benazia'])

    def test_catalan_sidenotes_removed(self):
        html = ('<div class="mw-parser-output"><p><span class="sidenote-right">Capitol primer</span>'
                'En nom de nostre senyor deus comença lo libre [f. 12 v.] que feu lo rey en Jacme.</p></div>')
        raw = json.dumps(dict(parse=dict(text=html, revid=1, title='T'))).encode()
        _, _, ps = v.ca_paragraphs(raw, dict(drop=None, sub=None))
        self.assertEqual([normalize_source(p) for p in ps],
                         ['en nom de nostre senyor deus comenca lo libre que feu lo rey en iacme'])


if __name__ == '__main__':
    unittest.main()

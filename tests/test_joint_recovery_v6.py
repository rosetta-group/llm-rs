import unittest
import experiments.joint_recovery_v6 as m


class PackTests(unittest.TestCase):
    def setUp(self):
        self.saved = m.MIN_LETTERS, m.MAX_LETTERS, m.PER_SOURCE
        m.MIN_LETTERS, m.MAX_LETTERS, m.PER_SOURCE = 10, 14, 2

    def tearDown(self):
        m.MIN_LETTERS, m.MAX_LETTERS, m.PER_SOURCE = self.saved

    def test_pack_builds_consecutive_blocks_and_skips_overlaps(self):
        rows = [dict(id=str(i), text=t) for i, t in enumerate(['abcde fg', 'hijk', 'bad row', 'lmnop qr', 'stuvw', 'xy'])]
        out = m.pack(rows, set(), 'd', 's')
        self.assertEqual([b['source_ids'] for b in out], [['0', '1'], ['2', '3']])
        m.MIN_LETTERS = 3
        rows = [dict(id='a', text=' '.join(['w'] * 20)), dict(id='b', text='abcd'), dict(id='c', text='efgh')]
        out = m.pack(rows, {tuple(['w'] * 20)}, 'd', 's')
        self.assertEqual([b['source_ids'] for b in out], [['b'], ['c']])
        with self.assertRaises(ValueError): m.pack(rows[:2], set(), 'd', 's')


if __name__ == '__main__':
    unittest.main()

import unittest
from experiments.length_scaling_v2 import scaled

BASE = dict(minimum=6, prune_usage=3., repair_minimum=2, repair_usage_floor=1., refine_cap=300, polish_cap=1200, repair_theta=5.)


class ScaledTests(unittest.TestCase):
    def test_identity_at_5200(self):
        for rule in ('linear', 'sqrt'):
            self.assertEqual(scaled(BASE, 5200, rule), BASE)

    def test_linear_and_sqrt_at_20800(self):
        a, b = scaled(BASE, 20800, 'linear'), scaled(BASE, 20800, 'sqrt')
        self.assertEqual((a['minimum'], a['prune_usage'], a['repair_minimum'], a['refine_cap']), (24, 12., 8, 1200.))
        self.assertEqual((b['minimum'], b['prune_usage'], b['repair_minimum'], b['refine_cap']), (12, 6., 4, 1200.))
        self.assertEqual(a['repair_theta'], 5.)


if __name__ == '__main__':
    unittest.main()

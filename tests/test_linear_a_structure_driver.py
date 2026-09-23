import unittest

from experiments.linear_a_structure_v2 import evaluate


class StructureDriverTest(unittest.TestCase):
    def test_control_score_includes_recalls_and_predictions(self):
        rows = [{'word': ('x'+str(i), 'a', 'b') if y else ('x'+str(i), 'b', 'a'),
                 'label': y, 'site': 'one', 'documents': ['doc'+str(i)]}
                for i in range(20) for y in (0,1)]
        result = evaluate(rows[:30], rows[30:], predictions=True)
        self.assertEqual(result['endings_class_recalls'], {'0': 1., '1': 1.})
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['predictions']), 10)

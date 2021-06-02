from unittest import TestCase


class TestExperiment(TestCase):
    def test_todict(self):
        from tools.utils import Experiment
        exp = Experiment({})
        d = exp.todict()
        self.assertEqual(len(d), 0)
        self.assertIsInstance(d, dict)

    def test_deepcopy(self):
        from copy import deepcopy
        from tools.utils import Experiment
        exp = Experiment({"abc": {"cde": 1, "eef": 2}, "uyu": 14, 2: True})
        exp = deepcopy(exp)

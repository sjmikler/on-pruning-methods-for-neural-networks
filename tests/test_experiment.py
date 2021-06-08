from unittest import TestCase


class TestExperiment(TestCase):
    def test_load_python_object(self):
        from tools.lazy_experiment import load_python_object
        import numpy as np

        assert load_python_object("numpy") == np
        assert load_python_object("numpy.random") == np.random
        assert load_python_object("numpy.random.rand") == np.random.rand
        assert isinstance(load_python_object("numpy.random.rand(10)"), np.ndarray)
        assert load_python_object("numpy.random.rand ") == np.random.rand
        assert load_python_object("numpy. random. rand") == np.random.rand

    def test_lazy_experiment(self):
        from tools import LazyExperiment
        from copy import deepcopy
        import numpy as np

        exp = {
            "t1": 2,
            "t2": 3,
            "nest": {"1": 2, "t3": "sparsity", "nest2": {"t4": 1, "t5": "solve 12"},},
            "t5": "solve 123",
            "arr": "solve numpy.random.rand()",
            "ref": "solve arr",
            "ref2": "solve self.arr",
            "new": ["solve 10", "solve t1", 15, "solve self.arr"]
        }
        exp["additional1"] = "solve numpy.random.rand(10)[:5]"
        exp["additional2"] = "solve numpy.random.rand(10) [:5]"
        exp["additional3"] = "solve numpy.random.rand (10) [:5]"
        exp["addon"] = "solve arr"

        exp = LazyExperiment(exp)
        assert exp.nest.nest2.t5 == 12
        assert exp.nest.nest2["t5"] != 12
        assert exp["nest"]["nest2"]["t5"] != 12
        assert exp.t5 == 123
        assert exp["t5"] != 123
        assert isinstance(exp.ref2, np.float)
        assert isinstance(exp.ref, str)
        exp.todict(solved=True)
        deepcopy(exp)

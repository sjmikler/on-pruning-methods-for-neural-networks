import pprint
import importlib
import re


def load_python_object(path, scope=None):
    scope = scope or {}
    scope = scope.copy()
    assert isinstance(path, str)
    try:
        func = eval(path, scope, scope)
        return func
    except (NameError, AttributeError) as e:
        try:
            module = importlib.import_module(path)
            return module
        except ModuleNotFoundError as e:
            if "." in path:
                split_path = re.split(r"[ ()\[\]/]", path)
                *module_path, _ = split_path[0].split(".")
                module_path = ".".join(module_path)
                remainder = path[len(module_path) :].lstrip(".")

                scope["__m__"] = importlib.import_module(module_path)
                func = eval(f"__m__.{remainder}", scope, scope)
                return func
            else:
                raise e


class LazyExperiment:
    SOLVE = "solve "

    def __init__(self, from_dict=None, parent=None):
        self.dict = {}
        self.parent = parent

        if from_dict:
            assert isinstance(from_dict, dict)
            for key, value in from_dict.items():
                self[key] = value
        self.solved = {}

    def _is_recipe(self, key=None):
        if isinstance(self.dict[key], list):
            return True
        if key not in self.dict or not isinstance(self.dict[key], str):
            return False
        return self.dict[key].startswith("solve ")

    def _get_object(self, key):
        if key in self.solved:
            return self.solved[key]
        else:
            solved = self._solve_from_recipe(self.dict[key])
            self.solved[key] = solved
            return self.solved[key]

    def _solve_from_recipe(self, recipe):
        if isinstance(recipe, str) and recipe.startswith(self.SOLVE):
            return load_python_object(
                recipe[len(self.SOLVE) :], scope={"self": self, **self.dict}
            )
        elif isinstance(recipe, list):
            for idx, value in enumerate(recipe):
                recipe[idx] = self._solve_from_recipe(value)
            return recipe
        else:
            return recipe

    def _reset_solved(self):
        self.solved = {}

    def __setitem__(self, key, value):
        if key in self.dict:
            raise KeyError(f"Overwritting existing key {key} is not permitted!")
        if isinstance(value, dict):
            value = LazyExperiment(value, parent=self)
        self.dict[key] = value

    def __getitem__(self, item):
        assert item in self.dict, f"Item {item} does not exitst!"
        return self.dict[item]

    def __getattr__(self, item):
        if item in self.__getattribute__("dict"):
            if self._is_recipe(item):
                return self._get_object(item)
            else:
                return self.dict[item]
        else:
            raise AttributeError  # If raised, python will execute __getattribute__

    def todict(self, solved=False):
        d = self.dict.copy()
        if solved:
            d = {k: self.__getattr__(k) for k in d}

        for k, v in d.items():
            if isinstance(v, LazyExperiment):
                d[k] = v.todict(solved=solved)
        return d

    def solve(self):
        solved = self.todict(solved=True)
        return LazyExperiment(solved)

    def keys(self):
        return self.dict.keys()

    def __repr__(self):
        d = self.todict()
        return pprint.pformat(d, sort_dicts=False)

    def __contains__(self, item):
        return self.dict.__contains__(item)

    def __iter__(self):
        return self.dict.__iter__()

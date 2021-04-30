import argparse
import os
import random
from collections.abc import Iterable
from copy import deepcopy

import yaml

from tools import utils

print = utils.get_cprint(color='yellow')


class YamlExperimentQueue:
    def __init__(self, experiments=None, path='.queue.yaml'):
        self.path = path
        self.num_popped = 0
        if experiments:  # if None, can just read existing experiments
            self.write_content(experiments)
        else:
            assert os.path.exists(path), "Neither experiments or queue were given!"
            raise NotImplementedError("UNTESTED!")

    def read_content(self):
        with open(self.path, 'r') as f:
            z = list(yaml.safe_load_all(f))
        return [utils.Experiment(exp) for exp in z]

    def write_content(self, exps):
        assert isinstance(exps, Iterable)

        with open(self.path, 'w') as f:
            yaml.safe_dump_all((exp.todict() for exp in exps),
                               stream=f,
                               explicit_start=True,
                               sort_keys=False)

    def append_content(self, exps):
        existing_content = self.read_content()
        exps = existing_content + exps
        self.write_content(exps)

    def __bool__(self):
        z = self.read_content()
        return bool(z)

    def pop(self):
        if self:  # else is empty
            exps = self.read_content()
        else:
            return None
        exp = exps.pop(0)
        self.write_content(exps)
        self.num_popped += 1
        return exp

    def __iter__(self):
        print(f"LOADING EXPERIMENT FROM {self.path}")
        while self:
            exp = self.pop()
            yield exp

    def __len__(self):
        return len(self.read_content()) + self.num_popped

    def close(self):
        os.remove(self.path)


def cool_parse_exp(exp, E, scopes=[]):
    keys = list(exp.keys())
    assert 'temp' not in keys
    assert 'E' not in keys

    for k in keys:
        v = exp[k]

        if isinstance(v, utils.Experiment):
            nscopes = deepcopy(scopes)
            nscopes.append(exp)
            parsed_v = cool_parse_exp(v, E, nscopes)
            exp[k] = parsed_v
            continue

        if isinstance(v, str) and v.startswith('eval'):
            org_expr = v
            v = v[4:].strip()

            scope = {}  # populating the scope for eval
            for new_scope in scopes:  # for each parent scope
                scope.update(new_scope)  # update current
            scope.update(deepcopy(exp))  # top it with this level scope
            scope['E'] = E  # and add experiment history

            v = eval(v, {}, scope)
            print(f"FANCY PARSING {k}: {org_expr} --> {v}")

        if isinstance(v, str):  # e.g. for parsing float in scientific notation
            try:
                v = float(v)
                v = int(v) if v == int(v) else v
            except ValueError:
                pass
        exp[k] = v
    return exp


def load_from_yaml(yaml_path, cmd_parameters=(), private_keys=()):
    parser = argparse.ArgumentParser(prefix_chars='+')
    for arg in cmd_parameters:
        if arg[0] == '+':
            if '=' in arg:
                arg = arg[:arg.index('=')]  # for usage with +arg=V
            parser.add_argument(arg)

    args = parser.parse_args(cmd_parameters)
    new_cmd_parameters = utils.Experiment()

    for key, value in args.__dict__.items():
        if isinstance(value, str):
            try:
                value = int(value) if float(value) == int(value) else float(value)
            except ValueError:
                pass
        new_cmd_parameters[key] = value

    print(f"CMD PARAMETERS: {new_cmd_parameters}")

    experiments = yaml.safe_load_all(open(yaml_path, "r"))
    experiments = [utils.Experiment(exp) for exp in experiments]
    default = experiments.pop(0)
    default.update(new_cmd_parameters)

    all_unpacked_experiments = []
    for global_rep in range(default.GlobalRepeat):
        unpacked_experiments = []
        for exp in experiments:
            nexp = deepcopy(exp)
            default_cpy = deepcopy(default)

            for key in nexp:
                if key in default_cpy:
                    default_cpy.pop(key)  # necessary to preserve order in dict
            nexp.update(default_cpy)

            for key in private_keys:
                if key in nexp:
                    nexp.pop(key)

            if "RND_IDX" in nexp:  # allow custom RND_IDX
                rnd_idx = nexp["RND_IDX"]
            else:
                rnd_idx = random.randint(100000, 999999)

            for rep in range(nexp.Repeat):
                nexp_rep = deepcopy(nexp)
                nexp_rep["RND_IDX"] = rnd_idx
                nexp_rep["REP"] = rep
                nexp_rep = cool_parse_exp(nexp_rep, unpacked_experiments)
                unpacked_experiments.append(nexp_rep)
        all_unpacked_experiments.extend(unpacked_experiments)

    if path := default.GlobalQueue:
        queue = YamlExperimentQueue(all_unpacked_experiments, path=path)
    else:
        queue = all_unpacked_experiments
    print(f"QUEUE TYPE: {type(queue)}")
    return default, queue

import os
import random
import socket
import sys
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
                v = eval(v, {}, {})
            except (NameError, SyntaxError):
                pass
        exp[k] = v
    return exp


def load_from_yaml(yaml_path, cmd_parameters=(), private_keys=()):
    experiments = yaml.safe_load_all(open(yaml_path, "r"))
    experiments = [utils.Experiment(exp) for exp in experiments]
    default = experiments.pop(0)

    assert 'Global' in default, "Global missing from default config!"

    parameters = [p for p in cmd_parameters if p.startswith('+')]
    print(f"FOUND CMD PARAMETERS: {parameters}")

    for cmd_param in parameters:
        try:
            sys.argv.remove(cmd_param)
            param = cmd_param.strip('+ ')
            param = 'default.' + param
            exec(param)
        except Exception as e:
            print(f"ERROR WHEN PARSING {cmd_param}!")
            raise e

    default.HOST = socket.gethostname()
    all_unpacked_experiments = []

    for global_rep in range(default.Global.repeat):
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
                rnd_idx = nexp.RND_IDX
            else:
                rnd_idx = random.randint(100000, 999999)

            for rep in range(nexp.Repeat):
                nexp_rep = deepcopy(nexp)
                nexp_rep.RND_IDX = rnd_idx
                nexp_rep.REP = rep
                nexp_rep = cool_parse_exp(nexp_rep, unpacked_experiments)
                unpacked_experiments.append(nexp_rep)
        all_unpacked_experiments.extend(unpacked_experiments)

    if path := default.Global.queue:
        queue = YamlExperimentQueue(all_unpacked_experiments, path=path)
    else:
        queue = all_unpacked_experiments
    print(f"QUEUE TYPE: {type(queue)}")
    return default, queue

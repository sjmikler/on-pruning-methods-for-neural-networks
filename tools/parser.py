import os
import random
from collections.abc import Iterable
from copy import deepcopy

import yaml

from tools.utils import cprint, ddict, unddict


class YamlExperimentQueue:
    def __init__(self, experiments=None, path='.queue.yaml'):
        self.path = path
        if experiments:  # if None, can just read existing experiments
            self.write_content(experiments)
        else:
            assert os.path.exists(path), "Neither experiments or queue were given!"

    def read_content(self):
        with open(self.path, 'r') as f:
            z = list(yaml.safe_load_all(f))
        return [ddict(exp) for exp in z]

    def write_content(self, exps):
        assert isinstance(exps, Iterable)
        with open(self.path, 'w') as f:
            nexps = map(unddict, exps)  # because cannot dump ddict
            yaml.safe_dump_all(nexps, stream=f)

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
        return exp

    def __iter__(self):
        cprint(f"LOADING EXPERIMENT FROM {self.path}")
        while self:
            exp = self.pop()
            yield exp

    def close(self):
        os.remove(self.path)


def cool_parse_exp(exp, E):
    keys = list(exp.keys())
    assert 'temp' not in keys
    assert 'E' not in keys

    for k in keys:
        v = exp[k]

        if isinstance(v, dict):
            parsed_v = cool_parse_exp(v, E)
            exp[k] = parsed_v
            continue

        if isinstance(v, str) and v.startswith('eval'):
            org_expr = v
            v = v[4:].strip()
            scope = deepcopy(exp)
            scope['E'] = E
            v = eval(v, {}, scope)
            cprint(f"RECOGNIZED FANCY PARSING {k}: {org_expr} --> {v}")

        if isinstance(v, str):
            try:
                v = float(v)
            except (ValueError, TypeError):
                pass
        exp[k] = v
    return exp


def load_from_yaml(yaml_path, unknown_args):
    cmd_arguments = ddict()
    for arg in unknown_args:
        cprint(f"COMMAND LINE ARGUMENT: {arg}")
        if '--' in arg and '=' in arg:
            key, value = arg.split('=', 1)
            key = key.lstrip('-')

            try:  # for parsing integers etc
                cmd_arguments[key] = eval(value, {}, {})
            except (NameError, SyntaxError):  # for parsing strings
                cmd_arguments[key] = value

    experiments = yaml.safe_load_all(open(yaml_path, "r"))
    experiments = [ddict(exp) for exp in experiments]
    default = experiments.pop(0)
    default.update(cmd_arguments)

    all_unpacked_experiments = []
    for global_rep in range(default.get("GLOBAL_REPEAT") or 1):
        unpacked_experiments = []
        for exp in experiments:
            nexp = deepcopy(default)
            nexp.update(exp)

            rnd_idx = random.randint(100000, 999999)
            for rep in range(exp.get("REPEAT") or 1):
                nexp_rep = deepcopy(nexp)
                nexp_rep['RND_IDX'] = rnd_idx
                nexp_rep['REP'] = rep

                nexp_rep = cool_parse_exp(nexp_rep, unpacked_experiments)
                unpacked_experiments.append(nexp_rep)
        all_unpacked_experiments.extend(unpacked_experiments)

    if path := default.queue:
        queue = YamlExperimentQueue(all_unpacked_experiments, path=path)
    else:
        queue = iter(all_unpacked_experiments)
    cprint(f'QUEUE TYPE: {type(queue)}')
    return default, queue

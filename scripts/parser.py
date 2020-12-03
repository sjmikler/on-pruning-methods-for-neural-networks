import random
import yaml
import sys
import os
import re
from copy import deepcopy


class YamlExperimentQueue:
    def __init__(self, experiments=None, path='.queue.yaml'):
        self.path = path
        if experiments:
            self.set_content(experiments)
        else:
            assert os.path.exists(path), "Neither experiments or queue was given!"

    def get_content(self):
        with open(self.path, 'r') as f:
            z = list(yaml.safe_load_all(f))
        return z

    def set_content(self, exps):
        with open(self.path, 'w') as f:
            if exps:
                yaml.safe_dump_all(exps, stream=f)

    def append_content(self, exps):
        existing_content = self.get_content()
        exps = existing_content + exps
        with open(self.path, 'w') as f:
            if exps:
                yaml.safe_dump_all(exps, stream=f)

    def __bool__(self):
        z = self.get_content()
        return bool(z)

    def pop(self):
        if self:
            exps = self.get_content()
        else:
            return None
        exp = exps.pop(0)
        self.set_content(exps)
        return exp

    def __iter__(self):
        print(f"LOADING EXPERIMENT FROM {self.path}")
        while self:
            exp = self.pop()
            yield exp

    def close(self):
        os.remove(self.path)


def cool_parse_exp(exp, past_experiments):
    keys = list(exp.keys())
    for k in keys:
        v = exp[k]
        if isinstance(v, dict):
            # recurrent parsing of inner dicts
            past_subdicts = [exp[k] if k in exp else {} for exp in past_experiments]
            exp[k] = cool_parse_exp(v, past_subdicts)
            continue

        cool_args = re.findall(r"([a-zA-Z0-9_]+)\[([-0-9]+)\]", str(v))
        for cool_name, cool_idx in cool_args:
            past_value = past_experiments[int(cool_idx)][cool_name]
            v = v.replace(f'{cool_name}[{cool_idx}]', str(past_value))
            print(
                f"REPLACED IN {k}: {cool_name}[{cool_idx}] WITH {past_value}")
        if isinstance(v, str) and v.startswith('exec'):
            expr1 = v
            v = v.replace('exec ', 'temp = ')
            exec_storage = exp
            exec(v, None, exec_storage)
            v = exec_storage.pop('temp')
            print(f"RECOGNIZED IN {k}: {expr1} AND REPLACED WITH {v}")
        if isinstance(v, str):
            try:
                v = float(v)
            except (ValueError, TypeError):
                pass
        exp[k] = v
    return exp


def load_from_yaml(yaml_path):
    cmd_arguments = {}
    for arg in sys.argv:
        print(f"COMMAND LINE ARGUMENT: {arg}")
        if '--' in arg and '=' in arg:
            key, value = arg.split('=', 1)
            key = key.lstrip('-')

            try:
                exec_storage = {}
                exec(f"temp = {value}", None, exec_storage)
                cmd_arguments[key] = exec_storage['temp']
            except (NameError, SyntaxError):  # for parsing strings
                cmd_arguments[key] = value

    experiments = list(yaml.safe_load_all(open(yaml_path, "r")))
    default = experiments.pop(0)
    default.update(cmd_arguments)

    unpacked_experiments = []
    for exp in experiments:
        expcp = exp.copy()
        exp.update(default)
        exp.update(expcp)
        rnd_idx = random.randint(100000, 999999)
        for rep in range(exp['REPEAT']):
            parsed_exp = deepcopy(exp)
            parsed_exp['IDX'] = f"{rnd_idx}/{rep}"

            parsed_exp = cool_parse_exp(parsed_exp, unpacked_experiments)
            unpacked_experiments.append(parsed_exp)

    if path := default.get('queue'):
        queue = YamlExperimentQueue(unpacked_experiments, path=path)
    else:
        queue = iter(unpacked_experiments)
    print(f'QUEUE TYPE: {type(queue)}')
    return default, queue

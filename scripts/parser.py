import random
import yaml
import sys
import os
import re
from copy import deepcopy


def cool_parse_exp(exp, past_experiments):
    for k, v in exp.items():
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
            v = v.replace('exec ', 'v = ')
            exec_storage = {}
            exec(v, exec_storage)
            v = exec_storage['v']
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
        if ':' in arg:
            key, value = arg.split(':', 1)

            d = {}
            exec(f"temp = {value}", None, d)
            cmd_arguments[key] = d['temp']

    experiments = list(yaml.safe_load_all(open(yaml_path, "r")))
    experiments[0].update(cmd_arguments)

    unpacked_experiments = []
    for exp in experiments[1:]:
        expcp = exp.copy()
        exp.update(experiments[0])
        exp.update(expcp)
        rnd_idx = random.randint(100000, 999999)
        for rep in range(exp['repeat']):
            parsed_exp = deepcopy(exp)
            parsed_exp['idx'] = f"{rnd_idx}/{rep}"
            path = f"{exp['directory']}/{exp['name']}/{parsed_exp['idx']}"
            parsed_exp['full_path'] = f"{path}"
            parsed_exp['checkpoint'] = f"{path}.h5"

            parsed_exp = cool_parse_exp(parsed_exp, unpacked_experiments)
            unpacked_experiments.append(parsed_exp)
    return experiments[0], unpacked_experiments


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
        while self:
            exp = self.pop()
            yield exp

    def close(self):
        os.remove(self.path)

# %%

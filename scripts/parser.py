import random
import yaml
import sys
import os


def load_from_yaml(yaml_path):
    cmd_arguments = {}
    for arg in sys.argv:
        if '--' in arg and ':' in arg:
            key, value = arg.split(':')

            d = {}
            exec(f"temp = {value}", None, d)
            cmd_arguments[key[2:]] = d['temp']

    experiments = list(yaml.safe_load_all(open(yaml_path, "r")))
    experiments[0].update(cmd_arguments)

    unpacked_experiments = []
    for exp in experiments[1:]:
        expcp = exp.copy()
        exp.update(experiments[0])
        exp.update(expcp)
        rnd_idx = random.randint(100000, 999999)
        for rep in range(exp['repeat']):
            exp['idx'] = f"{rnd_idx}/{rep}"
            unpacked_experiments.append(exp.copy())
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

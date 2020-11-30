import os
import yaml


class ddict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for key, value in self.items():
            if isinstance(value, dict):
                self.__setattr__(key, ddict(value))

    def __getattr__(self, key):
        if key in self:
            return self.get(key)
        else:
            return None

    def __setattr__(self, key, val):
        if isinstance(val, dict):
            val = ddict(val)

        if key in self.__dict__:
            self.__dict__[key] = val
        else:
            self[key] = val


def set_memory_growth():
    import tensorflow as tf
    gpu = tf.config.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu, True)


def disable_gpu():
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')


def contains_any(t, *opts):
    return any([x in t for x in opts])


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

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


def set_precision(precision):
    import tensorflow.keras.mixed_precision.experimental as mixed_precision

    if precision == 16:
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_policy(policy)


def disable_gpu():
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')


def contains_any(t, *opts):
    return any([x in t for x in opts])

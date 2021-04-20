from copy import deepcopy


class ddict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__['_unused_parameters'] = set()
        self.reset_unused_parameters()

        for key, value in self.items():
            if isinstance(value, dict):
                self.__setattr__(key, ddict(value))

    def __getattr__(self, key):
        if key in self:
            if key in self.__dict__['_unused_parameters']:
                self.__dict__['_unused_parameters'].remove(key)
            return self[key]
        else:
            return getattr(super(), key)

    def __setattr__(self, key, val):
        if isinstance(val, dict):
            val = ddict(val)
        self[key] = val
        self.__dict__['_unused_parameters'].add(key)

    def reset_unused_parameters(self, exclude=()):
        for k, v in self.items():
            if k in exclude:
                if k in self.__dict__['_unused_parameters']:
                    self.__dict__['_unused_parameters'].remove(k)
                continue
            self.__dict__['_unused_parameters'].add(k)

    def get_unused_parameters(self):
        return tuple(self.__dict__['_unused_parameters'])


def unddict(d):
    """Recursivly transform ddicts into dicts"""
    d = deepcopy(d)
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = unddict(v)
    return dict(d)


color2code = {
    'black': '\x1b[30m',
    'red': '\x1b[31m',
    'green': '\x1b[32m',
    'yellow': '\x1b[33m',
    'blue': '\x1b[34m',
    'white': '\x1b[37m',
    'reset': '\x1b[0m',
}


def cprint(*args, color='green', **kwargs):
    if color:
        print(color2code[color], end='')
    print(*args, **kwargs)
    if color:
        print(color2code['reset'], end='')


def set_memory_growth():
    import tensorflow as tf
    for gpu in tf.config.get_visible_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)


def set_visible_gpu(gpus=[]):
    import tensorflow as tf
    from collections.abc import Iterable
    if isinstance(gpus, Iterable):
        tf.config.set_visible_devices(gpus, 'GPU')
    else:
        tf.config.set_visible_devices([gpus], 'GPU')


def set_precision(precision):
    import tensorflow.keras.mixed_precision.experimental as mixed_precision

    if precision == 16:
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_policy(policy)


def contains_any(t, *opts):
    return any([x in t for x in opts])


def logging_from_history(history, info):
    import tensorflow as tf
    import datetime

    full_path = info["full_path"]
    writer = tf.summary.create_file_writer(full_path)
    cprint(f"FULL PATH: {full_path}")

    maxi_acc = max(history["val_accuracy"])
    date = datetime.datetime.now()
    info["TIME"] = f"{date.year}.{date.month}.{date.day} {date.hour}:{date.minute}"
    info["ACC"] = maxi_acc

    with writer.as_default():
        for key in history:
            for idx, value in enumerate(history[key]):
                tf.summary.scalar(key, value, idx + 1)
        tf.summary.text("experiment", data=str(info), step=0)

    with open(f"{info['yaml_logdir']}", "a") as f:
        for k, v in info.items():
            print(f"{k}: {v}", file=f)
        print("---", file=f)
    cprint(f"BEST ACCURACY: {maxi_acc}")


def parse(code):
    scope = {}
    exec(f'x = {code}', None, scope)
    return scope['x']


# noinspection PyUnresolvedReferences
def get_optimizer(optimizer, optimizer_config):
    import tensorflow as tf
    config = deepcopy(optimizer_config)
    optimizer = eval(optimizer)  # string -> optimizer

    for k, v in config.items():
        config[k] = eval(f"{config[k]}")
    return optimizer(**config)


def get_kernels(model):
    return [l.kernel for l in model.layers if hasattr(l, 'kernel')]


def set_all_weights_from_model(model, source_model):
    """Warning if a pair doesn't match."""

    for w1, w2 in zip(model.weights, source_model.weights):
        if w1.shape == w2.shape:
            w1.assign(w2)
        else:
            print(f"WARNING: Skipping {w1.name}: {w1.shape} != {w2.shape}")


def clone_model(model):
    """tf.keras.models.clone_model + toolkit.set_all_weights_from_model"""

    new_model = tf.keras.models.clone_model(model)
    set_all_weights_from_model(new_model, model)
    return new_model


def reset_weights_to_checkpoint(model, ckp=None, skip_keyword=None):
    """Reset network in place, has an ability to skip keybword."""
    import tensorflow as tf

    temp = tf.keras.models.clone_model(model)
    if ckp:
        temp.load_weights(ckp)
    skipped = 0
    for w1, w2 in zip(model.weights, temp.weights):
        if skip_keyword in w1.name:
            skipped += 1
            continue
        w1.assign(w2)
    cprint(f"INFO RESET: Skipped {skipped} layers with keyword {skip_keyword}!")
    return skipped


def clip_many(values, clip_at, clip_from=None, inplace=False):
    """Clips a list of tf or np arrays. Returns tf arrays."""
    import tensorflow as tf

    if clip_from is None:
        clip_from = -clip_at

    if inplace:
        for v in values:
            v.assign(tf.clip_by_value(v, clip_from, clip_at))
    else:
        r = []
        for v in values:
            r.append(tf.clip_by_value(v, clip_from, clip_at))
        return r


def concatenate_flattened(arrays):
    import numpy as np
    return np.concatenate([x.flatten() if isinstance(x, np.ndarray) else x.numpy(

    ).flatten() for x in arrays], axis=0)

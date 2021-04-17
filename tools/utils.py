from copy import deepcopy


class ddict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for key, value in self.items():
            if isinstance(value, dict):
                self.__setattr__(key, ddict(value))

    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            return getattr(super(), key)

    def __setattr__(self, key, val):
        if isinstance(val, dict):
            val = ddict(val)
        self[key] = val


def unddict(d):
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

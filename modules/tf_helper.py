import argparse
import sys
from collections import Counter
from collections.abc import Iterable
from copy import deepcopy

import tensorflow as tf

from tools.utils import get_cprint

cprint = get_cprint(color='light blue')


def set_memory_growth():
    cprint("SETTING MEMORY GROWTH!")
    for gpu in tf.config.get_visible_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)


def set_visible_gpu(gpus=[]):
    if isinstance(gpus, Iterable):
        tf.config.set_visible_devices(gpus, 'GPU')
    else:
        tf.config.set_visible_devices([gpus], 'GPU')


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--gpu",
                        default=None,
                        type=str,
                        help="Which GPUs to use during training, e.g. 0,1,3 or 1")
arg_parser.add_argument("--no-memory-growth",
                        action="store_true",
                        help="Disables memory growth")
args, unknown_args = arg_parser.parse_known_args()
if unknown_args:
    cprint(f"UNKNOWN CMD ARGUMENTS: {unknown_args}")
else:
    cprint(f"ALL CMD ARGUMENTS RECOGNIZED! (len(argv) = {len(sys.argv)})")

if args.gpu is not None:
    gpus = tf.config.get_visible_devices("GPU")
    gpu_indices = [num for num in range(10) if str(num) in args.gpu]
    cprint(f"SETTING VISIBLE GPUS TO {gpu_indices}")
    set_visible_gpu([gpus[idx] for idx in gpu_indices])

if not args.no_memory_growth:
    set_memory_growth()


def main(exp):
    cprint("RUNNING TENSORFLOW MODULE")
    set_precision(exp.precision)


# %%

# %%


def set_precision(precision):
    import tensorflow.keras.mixed_precision.experimental as mixed_precision

    cprint(f"SETTING PRECISION TO {precision}")
    if precision == 16:
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_policy(policy)


def logging_from_history(history, exp):
    import tensorflow as tf
    import datetime

    full_path = exp.full_path
    writer = tf.summary.create_file_writer(full_path)
    cprint(f"FULL PATH: {full_path}")

    maxi_acc = max(history["val_accuracy"])
    date = datetime.datetime.now()
    exp.TIME = f"{date.year}.{date.month}.{date.day} {date.hour}:{date.minute}"
    exp.ACC = maxi_acc

    with writer.as_default():
        for key in history:
            for idx, value in enumerate(history[key]):
                tf.summary.scalar(key, value, idx + 1)
        tf.summary.text("experiment", data=str(exp), step=0)

    with open(f"{exp.yaml_logdir}", "a") as f:
        for k, v in exp.items():
            print(f"{k}: {v}", file=f)
        print("---", file=f)
    cprint(f"BEST ACCURACY: {maxi_acc}")


def get_optimizer(optimizer, optimizer_config):
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
    return np.concatenate([x.flatten() if isinstance(x, np.ndarray)
                           else x.numpy().flatten() for x in arrays], axis=0)


def describe_model(model):
    cprint(f"MODEL INFO")
    layer_counts = Counter()
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer_counts['Dense'] += 1
        if isinstance(layer, tf.keras.layers.Conv2D):
            layer_counts['Conv2D'] += 1
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer_counts['BatchNorm'] += 1
        if isinstance(layer, tf.keras.layers.Dropout):
            layer_counts['Dropout'] += 1
    cprint(f"LAYER COUNTS: {dict(layer_counts)}")

    bn = 0
    biases = 0
    kernels = 0
    trainable_w = 0
    for w in model.trainable_weights:
        n = w.shape.num_elements()
        trainable_w += n

    for layer in model.layers:
        if hasattr(layer, 'beta') and layer.beta is not None:
            bn += layer.beta.shape.num_elements()

        if hasattr(layer, 'gamma') and layer.gamma is not None:
            bn += layer.gamma.shape.num_elements()

        if hasattr(layer, 'bias') and layer.bias is not None:
            biases += layer.bias.shape.num_elements()

        if hasattr(layer, 'kernel'):
            kernels += layer.kernel.shape.num_elements()

    cprint(f"TRAINABLE WEIGHTS: {trainable_w}")
    cprint(f"KERNELS: {kernels} ({kernels / trainable_w * 100:^6.2f}%), "
           f"BIASES: {biases} ({biases / trainable_w * 100:^6.2f}%), "
           f"BN: {bn} ({bn / trainable_w * 100:^6.2f}%)")

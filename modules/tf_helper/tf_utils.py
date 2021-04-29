from collections import Counter, abc
from copy import deepcopy

import numpy as np
import tensorflow as tf

from ._initialize import *


def set_memory_growth():
    print("SETTING MEMORY GROWTH!")
    for gpu in tf.config.get_visible_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)


def set_visible_gpu(gpus=[]):
    if isinstance(gpus, abc.Iterable):
        tf.config.set_visible_devices(gpus, 'GPU')
    else:
        tf.config.set_visible_devices([gpus], 'GPU')


def set_precision(precision):
    import tensorflow.keras.mixed_precision.experimental as mixed_precision

    print(f"SETTING PRECISION TO {precision}")
    if precision == 16:
        policy = mixed_precision.Policy("mixed_float16")
    elif precision == 32:
        policy = mixed_precision.Policy('float32')
    elif precision == 64:
        policy = mixed_precision.Policy('float64')
    mixed_precision.set_policy(policy)


def logging_from_history(history, exp):
    import datetime
    import socket

    min_loss = min(history["val_loss"])
    max_acc = max(history["val_accuracy"])
    final_acc = history["val_accuracy"][-1]

    min_tr_loss = min(history["loss"])
    max_tr_acc = max(history["accuracy"])

    print(f"FULL PATH: {exp.full_path}")
    print(f"BEST ACCURACY: {max_acc}")

    exp.TIME = datetime.datetime.now().strftime("%Y.%m.%d %H:%M")
    exp.HOST = socket.gethostname()
    exp.ACC = max_acc
    exp.FINAL_ACCU = final_acc
    exp.VALID_LOSS = min_loss
    exp.TRAIN_ACCU = max_tr_acc
    exp.TRAIN_LOSS = min_tr_loss

    writer = tf.summary.create_file_writer(exp.full_path)
    with writer.as_default():
        for key in history:
            for idx, value in enumerate(history[key]):
                tf.summary.scalar(key, value, idx + 1)
        tf.summary.text("experiment", data=str(exp), step=0)
    return exp


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

    temp = tf.keras.models.clone_model(model)
    if ckp:
        temp.load_weights(ckp)
    skipped = 0
    for w1, w2 in zip(model.weights, temp.weights):
        if skip_keyword in w1.name:
            skipped += 1
            continue
        w1.assign(w2)
    print(f"INFO RESET: Skipped {skipped} layers with keyword {skip_keyword}!")
    return skipped


def clip_many(values, clip_at, clip_from=None, inplace=False):
    """Clips a list of tf or np arrays. Returns tf arrays."""

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
    return np.concatenate([x.flatten() if isinstance(x, np.ndarray)
                           else x.numpy().flatten() for x in arrays], axis=0)


def describe_model(model):
    print(f"MODEL INFO")
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
    print(f"LAYER COUNTS: {dict(layer_counts)}")

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

    print(f"TRAINABLE WEIGHTS: {trainable_w}")
    print(f"KERNELS: {kernels} ({kernels / trainable_w * 100:^6.2f}%), "
          f"BIASES: {biases} ({biases / trainable_w * 100:^6.2f}%), "
          f"BN: {bn} ({bn / trainable_w * 100:^6.2f}%)")

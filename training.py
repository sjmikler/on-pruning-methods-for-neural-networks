# %%

import datetime
import re
import time
import numpy as np
import tensorflow as tf
import yaml
from tensorflow.keras.mixed_precision import experimental as mixed_precision

import os
from tools.utils import ddict, set_memory_growth, YamlExperimentQueue

set_memory_growth()

from tools import datasets, layers, models, pruning

experiments = list(yaml.safe_load_all(
    open("train/experiment.yaml", "r")))

# Create YamlExperimentQueue
unpacked_experiments = []
for exp in experiments[1:]:
    expcp = exp.copy()
    exp.update(experiments[0])
    exp.update(expcp)
    rnd_idx = np.random.randint(10000, 100000)
    for rep in range(exp['repeat']):
        exp['idx'] = f"{rnd_idx}/{rep}"
        unpacked_experiments.append(exp.copy())

experiment_queue = YamlExperimentQueue(unpacked_experiments, path='queue.yaml')
default_config = ddict(experiments[0])

if default_config.precision == 16:
    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_policy(policy)


# %%

def log_from_history(history, info):
    full_path = info['full_path']
    writer = tf.summary.create_file_writer(f"{full_path}")
    print(f"FULL PATH: {full_path}")

    with writer.as_default():
        for key in history:
            for idx, value in enumerate(history[key]):
                tf.summary.scalar(key, value, idx + 1)
        tf.summary.text("experiment", data=str(exp), step=0)

    maxi_acc = max(history["val_accuracy"])
    with open(f"{info['directory']}/{info['name']}/log.yaml", "a") as f:
        date = datetime.datetime.now()
        info["time"] = f"{date.year}.{date.month}.{date.day} {date.hour}:{date.minute}"
        info["acc"] = maxi_acc
        for k, v in info.items():
            print(f"{k}: {v}", file=f)
        print("---", file=f)
    print(f"BEST ACCURACY: {maxi_acc}", end="\n\n")


def get_optimizer(boundaries, values):
    schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values)
    return tf.keras.optimizers.SGD(learning_rate=schedule, momentum=0.9, nesterov=True)


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
ds = datasets.get_dataset(default_config.dataset, default_config.precision)

# %%

past_experiments = []


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


# %%

for exp in experiment_queue:
    exp = ddict(cool_parse_exp(exp, past_experiments))
    path = f"{exp.directory}/{exp.name}/{exp.idx}"
    exp.full_path = f"{path}"
    exp.checkpoint = f"{path}.h5"
    print('EXPERIMENT:', str(exp))

    # PROCEDURES IN ORDER:
    # 1. Creating model
    # 2. Loading checkpoint Before Pruning
    # 3. Applying pruning
    # 4. Loading checkpoint After Pruning
    # 5. Pruning related procedures After Pruning

    optimizer = get_optimizer(exp.lr_boundaries, exp.lr_values)
    model_func = models.get_model(exp.model)
    model_config = exp.model_config
    model_config.l2_reg = float(model_config.l2_reg)
    model = model_func(**model_config)

    # load checkpointed weights before the pruning
    if exp.checkpointBP:
        model.load_weights(exp.checkpointBP)
        print(f"LOADED BEFORE PRUNING {exp.checkpointBP}")

    # calculate pruning masks for network
    model = pruning.set_pruning_masks(
        model=model,
        pruning_method=exp.pruning,
        pruning_config=exp.pruning_config,
        dataset=ds
    )
    assert isinstance(model, tf.keras.Model)

    # load or reset weights after the pruning, do not change masks
    if exp.checkpointAP is not None:
        model_copy = model
        model = model_func(**model_config)
        if exp.checkpointAP != "random":
            model.load_weights(exp.checkpointAP)
        num_masks = 0
        for w1, w2 in zip(model_copy.weights, model.weights):
            if "kernel_mask" in w1.name:
                w2.assign(w1.value())
                num_masks += 1
        del model_copy
        for layer in model.layers:
            if hasattr(layer, "apply_pruning_mask"):
                layer.apply_pruning_mask()
        print(
            f"LOADED AFTER PRUNING {exp.checkpointAP}, but keeping {num_masks} masks!")

    # apply pruning from previously calculated masks
    pruning.apply_pruning_masks(model, pruning_method=exp.pruning)
    model.compile(optimizer, loss_fn, metrics=["accuracy"])

    if exp.name == "skip":  # just skip the training
        print("SKIPPING TRAINING", end="\n\n")
    else:
        steps_per_epoch = min(exp.num_iterations,
                              exp.steps_per_epoch)  # for short trainings

        try:
            history = model.fit(
                x=ds.train,
                validation_data=ds.valid,
                steps_per_epoch=steps_per_epoch,
                epochs=int(exp.num_iterations / steps_per_epoch),
            )
            log_from_history(history.history, info=exp.copy())
            model.save_weights(exp.checkpoint, save_format="h5")
        except KeyboardInterrupt:
            print("SKIPPING EXPERIMENT, WAITING 2 SECONDS TO RESUME...")
            time.sleep(2)
    past_experiments.append(exp)
experiment_queue.close()

# %%

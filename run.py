# %%

import datetime
import pprint
import sys
import time

import tensorflow as tf

from tools import datasets, models, parser, pruning
from tools import utils
from tools.utils import ddict

utils.set_memory_growth()

default_config, experiment_queue = parser.load_from_yaml(yaml_path="experiment.yaml")
default_config = ddict(default_config)
utils.set_precision(default_config.precision)

if "--dry" in sys.argv:
    dry = True
else:
    dry = False


# %%


def log_from_history(history, model, info):
    full_path = info["full_path"]
    writer = tf.summary.create_file_writer(f"{full_path}")
    print(f"FULL PATH: {full_path}")

    with writer.as_default():
        for key in history:
            for idx, value in enumerate(history[key]):
                tf.summary.scalar(key, value, idx + 1)
        tf.summary.text("experiment", data=str(exp), step=0)

    maxi_acc = max(history["val_accuracy"])
    date = datetime.datetime.now()
    info["DENSITY"] = pruning.report_density(model)
    info["TIME"] = f"{date.year}.{date.month}.{date.day} {date.hour}:{date.minute}"
    info["ACC"] = maxi_acc

    with open(f"{info['yaml_logdir']}", "a") as f:
        for k, v in info.items():
            print(f"{k}: {v}", file=f)
        print("---", file=f)
    print(f"BEST ACCURACY: {maxi_acc}", end="\n\n")


def get_optimizer(boundaries, values):
    schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    return tf.keras.optimizers.SGD(learning_rate=schedule, momentum=0.9, nesterov=True)


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
ds = datasets.get_dataset(default_config.dataset, default_config.precision)

# %%

for exp in experiment_queue:
    print("\nEXPERIMENT:")
    pprint.pprint(exp)
    print()
    if dry:
        continue
    exp = ddict(exp)

    # PROCEDURES IN ORDER:
    # 1. Creating model
    # 2. Loading checkpoint Before Pruning
    # 3. Applying pruning
    # 4. Loading checkpoint After Pruning
    # 5. Pruning related procedures After Pruning

    optimizer = get_optimizer(exp.lr_boundaries, exp.lr_values)
    model_func = models.get_model(exp.model)
    model_config = exp.model_config
    model_config.l1_reg = float(model_config.l1_reg)
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
        dataset=ds,
    )
    assert isinstance(model, tf.keras.Model)

    # load or reset weights after the pruning, do not change masks
    if exp.checkpointAP is not None:
        if exp.checkpointAP == 'random':
            ckp = None
        else:
            ckp = exp.checkpointAP
        num_masks = pruning.reset_weights_to_checkpoint(model, ckp=ckp, skip_keyword='kernel_mask')
        print(f"LOADED AFTER PRUNING {exp.checkpointAP}, but keeping {num_masks} masks!")

    # apply pruning from previously calculated masks
    pruning.apply_pruning_masks(model, pruning_method=exp.pruning)
    model.compile(optimizer, loss_fn, metrics=["accuracy"])

    if exp.name == "skip":  # just skip the training
        print("SKIPPING TRAINING", end="\n\n")
    else:
        steps_per_epoch = min(exp.num_iterations, exp.steps_per_epoch)
        try:
            history = model.fit(
                x=ds.train,
                validation_data=ds.valid,
                steps_per_epoch=steps_per_epoch,
                epochs=int(exp.num_iterations / steps_per_epoch),
            )
            log_from_history(history.history, model, info=exp.copy())
            model.save_weights(exp.checkpoint, save_format="h5")

        except KeyboardInterrupt:
            print("SKIPPING EXPERIMENT, WAITING 2 SECONDS BEFORE RESUMING...")
            time.sleep(2)

if isinstance(experiment_queue, parser.YamlExperimentQueue):
    experiment_queue.close()

# %%

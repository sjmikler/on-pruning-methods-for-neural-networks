import argparse
import pprint
import time
from copy import deepcopy

from tools.utils import cprint, ddict, logging_from_history

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--dry", action='store_true',
                        help="Skip training but parse experiments")
arg_parser.add_argument("--no-memory-growth", action='store_true',
                        help="Disables memory growth")
arg_parser.add_argument("--gpu", default=None, type=str,
                        help="Which GPUs to use during training, e.g. 0,1,3 or 1")
arg_parser.add_argument("--pick", "--cherrypick-experiments", default=None, type=str,
                        help="Run only selected experiments, e.g. 0,1,3 or 1")
args, unknown_args = arg_parser.parse_known_args()

import tensorflow as tf
from tools import datasets, parser, pruning, utils, models

if args.gpu is not None:
    gpus = tf.config.get_visible_devices('GPU')
    gpu_indices = [num for num in range(10) if str(num) in args.gpu]
    gpus = [gpus[idx] for idx in gpu_indices]
    tf.config.set_visible_devices(gpus, 'GPU')

if not args.no_memory_growth:
    utils.set_memory_growth()

default_config, experiment_queue = parser.load_from_yaml(yaml_path="experiment.yaml",
                                                         unknown_args=unknown_args)
utils.set_precision(default_config.precision)


# %%

def get_optimizer(optimizer, optimizer_config):
    config = deepcopy(optimizer_config)
    optimizer = eval(optimizer, None, {})  # string -> optimizer

    for k, v in config.items():
        config[k] = eval(f"{config[k]}", None, {})
    return optimizer(**config)


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
ds = datasets.get_dataset(default_config.dataset,
                          precision=default_config.precision,
                          **default_config.dataset_config)

# %%

for exp_idx, exp in enumerate(experiment_queue):
    if args.pick is not None:
        if str(exp_idx) not in args.pick:
            cprint(f"SKIPPING EXPERIMENT {exp_idx}")
            continue

    cprint("EXPERIMENT:")
    pprint.pprint(exp)
    print()
    if args.dry:
        continue
    exp = ddict(exp)

    # PROCEDURES IN ORDER:
    # 1. Creating model
    # 2. Loading checkpoint Before Pruning
    # 3. Applying pruning
    # 4. Loading checkpoint After Pruning
    # 5. Pruning related procedures After Pruning

    optimizer = get_optimizer(exp.optimizer, exp.optimizer_config)
    model_func = models.get_model(exp.model)
    model_config = exp.model_config
    model = model_func(**model_config)
    # model.summary()

    # load checkpointed weights before the pruning
    if hasattr(exp, 'checkpointBP'):
        model.load_weights(exp.checkpointBP)
        cprint(f"LOADED BEFORE PRUNING {exp.checkpointBP}")

    # calculate pruning masks for network
    model = pruning.set_pruning_masks(
        model=model,
        pruning_method=exp.pruning,
        pruning_config=exp.pruning_config,
        dataset=ds,
    )
    assert isinstance(model, tf.keras.Model)

    # load or reset weights after the pruning, do not change masks
    if hasattr(exp, 'checkpointAP'):
        if exp.checkpointAP == 'random':
            ckp = None
        else:
            ckp = exp.checkpointAP
        num_masks = pruning.reset_weights_to_checkpoint(model, ckp=ckp,
                                                        skip_keyword='kernel_mask')
        cprint(
            f"LOADED AFTER PRUNING {exp.checkpointAP}, but keeping {num_masks} masks!")

    # apply pruning from previously calculated masks
    pruning.apply_pruning_masks(model, pruning_method=exp.pruning)
    model.compile(optimizer, loss_fn, metrics=["accuracy"])

    if exp.name == "skip":  # just skip the training
        cprint("SKIPPING TRAINING")
    else:
        steps_per_epoch = min(exp.num_iterations, exp.steps_per_epoch)
        try:
            history = model.fit(
                x=ds.train,
                validation_data=ds.valid,
                steps_per_epoch=steps_per_epoch,
                epochs=int(exp.num_iterations / steps_per_epoch),
            )
            info = exp.copy()
            info["FINAL_DENSITY"] = pruning.report_density(model)
            logging_from_history(history.history, info=info)
            model.save_weights(exp.checkpoint, save_format="h5")

        except KeyboardInterrupt:
            cprint("\n\nSKIPPING EXPERIMENT, WAITING 2 SECONDS BEFORE RESUMING...")
            time.sleep(2)

if isinstance(experiment_queue, parser.YamlExperimentQueue):
    cprint(f"REMOVING QUEUE {experiment_queue.path}")
    experiment_queue.close()

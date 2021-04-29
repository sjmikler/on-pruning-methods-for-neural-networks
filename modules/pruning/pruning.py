import os

import tensorflow as tf

from modules import tf_helper
from modules.tf_helper import datasets, models, tf_utils
from . import pruning_utils
from ._initialize import *


def main(exp):
    """
    PROCEDURES IN ORDER:
    1. Loading inherited module - tf_utils
    2. Creating dataset, model, optimizer
    3. Loading checkpoint Before Pruning
    4. Applying pruning
    5. Loading checkpoint After Pruning
    6. Pruning related procedures After Pruning
    7. Training

    EXPERIMENT KEYS:
    * checkpointBP: not required
    * checkpointAP: not required
    * ...
    """
    print("RUNNING PRUNING MODULE")

    tf_helper.main(exp)  # RUN INHERITED MODULES

    pruning_utils.globally_enable_pruning()

    ds = datasets.get_dataset(exp.dataset,
                              precision=exp.precision,
                              **exp.dataset_config)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model_func = models.get_model(exp.model)
    model_config = exp.model_config
    model = model_func(**model_config)

    optimizer = tf_utils.get_optimizer(exp.optimizer, exp.optimizer_config)
    model.compile(optimizer, loss_fn, metrics=["accuracy"])
    tf_utils.describe_model(model)

    # load checkpointed weights before the pruning
    if hasattr(exp, 'checkpointBP'):
        model.load_weights(exp.checkpointBP)
        print(f"LOADED BEFORE PRUNING {exp.checkpointBP}")

    model = pruning_utils.set_pruning_masks(model=model,
                                            pruning_method=exp.pruning,
                                            pruning_config=exp.pruning_config,
                                            dataset=ds)
    assert isinstance(model, tf.keras.Model)

    # load or reset weights after the pruning, do not change masks
    if hasattr(exp, 'checkpointAP'):
        if exp.checkpointAP == 'random':
            ckp = None
        else:
            ckp = exp.checkpointAP
        num_masks = tf_utils.reset_weights_to_checkpoint(model,
                                                         ckp=ckp,
                                                         skip_keyword='kernel_mask')
        print(f"LOADED AFTER PRUNING {exp.checkpointAP}, but keeping {num_masks} "
              f"masks!")

    # apply pruning from previously calculated masks
    pruning_utils.apply_pruning_masks(model, pruning_method=exp.pruning)
    steps_per_epoch = min(exp.steps, exp.steps_per_epoch)

    if unused := exp.get_unused_parameters():
        print("WARNING! Unused parameters:")
        print(unused)

    if exp.steps != 0:
        history = model.fit(x=ds['train'],
                            validation_data=ds['valid'],
                            steps_per_epoch=steps_per_epoch,
                            epochs=int(exp.steps / steps_per_epoch))
        exp.FINAL_DENSITY = pruning_utils.report_density(model)
        print("FINAL DENSITY:", exp.FINAL_DENSITY)
        tf_utils.logging_from_history(history.history, exp=exp)

    if dirpath := os.path.dirname(exp.checkpoint):
        os.makedirs(dirpath, exist_ok=True)
    model.save_weights(exp.checkpoint, save_format="h5")


if __name__ == '__main__':
    exp = {
        'precision': 16,
        'name': 'temp',
        'full_path': 'temp',
        'checkpoint': 'temp.h5',
        'yaml_logdir': 'temp.yaml',
        'steps': 200,
        'steps_per_epoch': 20,
        'dataset': 'cifar10',
        'dataset_config': {
            'train_batch_size': 128,
            'valid_batch_size': 512,
        },
        'optimizer': 'tf.optimizers.SGD',
        'optimizer_config': {
            'learning_rate': 0.1,
            'momentum': 0.9,
        },
        'pruning': 'magnitude',
        'pruning_config': {
            'sparsity': 0.5,
        },
        'model': 'lenet',
        'model_config': {
            'input_shape': [32, 32, 3],
            'n_classes': 10,
            'l2_reg': 1e-5,
        }
    }
    # exp = utils.ddict(exp)
    # main(exp)

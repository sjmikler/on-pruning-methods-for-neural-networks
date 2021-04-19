import tensorflow as tf

from tools import datasets, models, pruning, pruning_toolkit
from tools.utils import cprint, get_optimizer, logging_from_history


def main(exp):
    """
    PROCEDURES IN ORDER:
    1. Creating model
    2. Loading checkpoint Before Pruning
    3. Applying pruning
    4. Loading checkpoint After Pruning
    5. Pruning related procedures After Pruning

    EXPERIMENT KEYS:
    * checkpointBP: not required
    * checkpointAP: not required
    """
    cprint("RUNNING PRUNING MODULE! TRAINING WILL START...")

    ds = datasets.get_dataset(exp.dataset,
                              precision=exp.precision,
                              **exp.dataset_config)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model_func = models.get_model(exp.model)
    model_config = exp.model_config
    model = model_func(**model_config)

    optimizer = get_optimizer(exp.optimizer, exp.optimizer_config)
    model.compile(optimizer, loss_fn, metrics=["accuracy"])

    # load checkpointed weights before the pruning
    if hasattr(exp, 'checkpointBP'):
        model.load_weights(exp.checkpointBP)
        cprint(f"LOADED BEFORE PRUNING {exp.checkpointBP}")

    pruning.globally_enable_pruning()
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

    kernels = pruning_toolkit.get_kernels(model)
    kernel_masks = pruning_toolkit.get_kernel_masks(model)

    class Callback(tf.keras.callbacks.Callback):
        def on_train_batch_begin(self, batch, logs=None):
            for k, km in zip(kernels, kernel_masks):
                k.assign(k * km)

    steps_per_epoch = min(exp.steps, exp.steps_per_epoch)
    history = model.fit(
        x=ds.train,
        validation_data=ds.valid,
        steps_per_epoch=steps_per_epoch,
        epochs=int(exp.steps / steps_per_epoch),
        callbacks=[Callback()]
    )
    info = exp.copy()
    info["FINAL_DENSITY"] = pruning.report_density(model)
    logging_from_history(history.history, info=info)
    model.save_weights(exp.checkpoint, save_format="h5")

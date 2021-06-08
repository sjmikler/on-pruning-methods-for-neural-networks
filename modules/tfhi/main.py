import argparse

import tensorflow as tf

from modules.tfhi import datasets, logging, models, tools, training_functools

try:
    from ._initialize import *
except ImportError:
    pass

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--gpu",
    type=int,
    nargs="*",
    help="Which GPUs to use during training, e.g. 0 1 3 or just 1",
)
arg_parser.add_argument(
    "--no-memory-growth", action="store_true", help="Disables memory growth"
)
args, unknown_args = arg_parser.parse_known_args()

print(f"UNKNOWN CMD ARGUMENTS: {unknown_args}")
print(f"  KNOWN CMD ARGUMENTS: {args.__dict__}")

if args.gpu:
    gpus = tf.config.get_visible_devices("GPU")
    print(f"SETTING VISIBLE GPUS TO {args.gpu}")
    tools.set_visible_gpu([gpus[idx] for idx in args.gpu])

if not args.no_memory_growth:
    tools.set_memory_growth()


def main(exp):
    print("RUNNING TFHI MODULE")
    tools.set_precision(exp.precision)

    optimizer = exp.optimizer

    if isinstance(exp.loss_fn, str):
        loss_fn = tools.get_loss_fn_from_alias(exp.loss_fn)
    else:
        loss_fn = exp.loss_fn

    if isinstance(exp.dataset, str):
        dataset = datasets.get_dataset_from_alias(exp.dataset, exp.precision)
    else:
        dataset = exp.dataset

    if isinstance(exp.model, str):
        model = models.get_model_from_alias(
            exp.model,
            input_shape=datasets.figure_out_input_shape(dataset),
            n_classes=datasets.figure_out_n_classes(dataset),
        )
    else:
        model = exp.model
    assert isinstance(model, tf.keras.Model)

    metrics = ["accuracy"]

    lr_metric = tools.get_optimizer_lr_metric(optimizer)
    if lr_metric:
        metrics.append(lr_metric)

    model.compile(optimizer, loss_fn, metrics=metrics)
    tools.print_model_info(model)

    steps_per_epoch = exp.steps_per_epoch

    if hasattr(exp, "epochs"):
        num_epochs = exp.epochs
    elif hasattr(exp, "steps"):
        if exp.steps < steps_per_epoch:
            steps_per_epoch = exp.steps
        num_epochs = int(exp.steps / steps_per_epoch)
    else:
        num_epochs = 0

    if hasattr(exp, "initial_epoch"):
        initial_epoch = exp.initial_epoch
    else:
        initial_epoch = 0

    if hasattr(exp, "callbacks"):
        callbacks = exp.callbacks
        for callback in callbacks:
            assert isinstance(callback, tf.keras.callbacks.Callback)
            callback.set_model(model)
    else:
        callbacks = []

    if num_epochs > initial_epoch:
        if hasattr(exp, "custom_training") and exp.custom_training:
            history = training_functools.fit(
                model=model,
                training_data=dataset["train"],
                validation_data=dataset["test"],
                steps_per_epoch=steps_per_epoch,
                epochs=num_epochs,
                initial_epoch=initial_epoch,
                callbacks=callbacks,
            )
        else:
            history = model.fit(
                x=dataset["train"],
                validation_data=dataset["test"],
                steps_per_epoch=steps_per_epoch,
                epochs=num_epochs,
                initial_epoch=initial_epoch,
                callbacks=callbacks,
            ).history

        logging.log_from_history(history, exp=exp)


if __name__ == "__main__":

    class Exp:
        name = "temp"
        precision = 16
        save_model = {}
        save_optim = {}
        tensorboard_log = None
        steps = 200
        steps_per_epoch = 20
        model = "VGG13"
        dataset = "cifar10"
        optimizer = tf.optimizers.SGD(0.1)
        loss_fn = "crossentropy"
        pruning = "magnitude"
        pruning_config = {"sparsity": 0.5}

    main(Exp)

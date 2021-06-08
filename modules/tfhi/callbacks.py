import tensorflow as tf

from modules.tfhi import pruning, tools
import tqdm

try:
    from ._initialize import *
except ImportError:
    pass


class Load(tf.keras.callbacks.Callback):
    def __init__(self, path, optimizer=False, skip_keyword=None):
        super().__init__()
        self.path = path
        self.optimizer = optimizer
        self.skip_keyword = skip_keyword

    def on_train_begin(self, logs=None):
        num_skipped = tools.reset_weights_to_checkpoint(
            self.model, ckp=self.path, skip_keyword=self.skip_keyword
        )
        if num_skipped:
            print(f"LOADED {self.path}, but keeping {num_skipped} weights intact!")
        else:
            print(f"LOADED {self.path}")

        if self.optimizer:
            tools.build_optimizer(self.model, self.model.optimizer)
            tools.update_optimizer(self.model.optimizer, self.optimizer)
            print(f"LOADED OPTIMIZER {self.optimizer}")


class CheckpointAfterEpoch(tf.keras.callbacks.Callback):
    def __init__(self, epoch2path, epoch2path_optim):
        super().__init__()
        self.epoch2path = epoch2path
        self.epoch2path_optim = epoch2path_optim
        self.created_model_ckp = []
        self.created_optim_ckp = []

    def on_train_begin(self, logs=None):
        self.on_epoch_end(-1)

    def on_epoch_end(self, epoch, logs=None):
        next_epoch = epoch + 1

        if next_epoch in self.epoch2path:
            path = self.epoch2path[next_epoch]
            tools.save_model(self.model, path)
            self.created_model_ckp.append(path)

        if next_epoch in self.epoch2path_optim:
            path = self.epoch2path_optim[next_epoch]
            tools.save_optimizer(self.model.optimizer, path)
            self.created_optim_ckp.append(path)

    def on_train_end(self, logs=None):
        print(f"CREATED MODEL CHECKPOINTS:")
        for ckp in self.created_model_ckp:
            print(ckp)
        print(f"CREATED OPTIM CHECKPOINTS:")
        for ckp in self.created_optim_ckp:
            print(ckp)


def save_and_report_density(self, logs=None):
    density = pruning.report_density(self.model, silent=True)
    if self.exp is not None:
        self.exp["FINAL_DENSITY"] = density
    print(f"FINAL DENSITY: {density}")


class OneShotPruning(tf.keras.callbacks.Callback):
    def __init__(self, pruning, pruning_config, dataset=None, exp=None):
        super().__init__()
        self.pruning = pruning
        self.pruning_config = pruning_config
        self.dataset = dataset
        self.exp = exp

    def on_train_begin(self, logs=None):
        model = pruning.set_pruning_masks(
            model=self.model,
            pruning_method=self.pruning,
            pruning_config=self.pruning_config,
            dataset=self.dataset,
        )
        pruning.apply_pruning_for_model(model)

    def on_train_end(self, logs=None):
        save_and_report_density(self, logs)


class CosinePruning(tf.keras.callbacks.Callback):
    def __init__(
        self, decay_steps, alpha, interval=100, verbose_interval=2000, exp=None
    ):
        super().__init__()
        self.schedule = tf.keras.experimental.CosineDecay(1.0, decay_steps, alpha)
        self.step = 0
        self.interval = interval
        self.verbose_interval = verbose_interval
        self.exp = exp

    def on_train_batch_begin(self, batch, logs=None):
        if batch % self.interval == 0:
            density = self.schedule(self.step)
            model = pruning.prune_l1(
                model=self.model, config={"sparsity": 1 - density}, silent=True
            )
            silent = (self.step % self.verbose_interval) != 0
            if not silent:
                density = pruning.report_density(model, silent=True)
                tqdm.tqdm.write(f"REPORTED DENSITY: {density}")
            pruning.apply_pruning_for_model(model)
            self.step += self.interval

    def on_train_end(self, logs=None):
        save_and_report_density(self, logs)


class PolynomialPruning(tf.keras.callbacks.Callback):
    def __init__(
        self, decay_steps, alpha, interval=100, verbose_interval=2000, exp=None
    ):
        super().__init__()
        self.schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            1.0, decay_steps, end_learning_rate=alpha, power=3.0
        )
        self.step = 0
        self.interval = interval
        self.verbose_interval = verbose_interval
        self.exp = exp

    def on_train_batch_begin(self, batch, logs=None):
        if batch % self.interval == 0:
            density = self.schedule(self.step)
            model = pruning.prune_l1(
                model=self.model, config={"sparsity": 1 - density}, silent=True
            )
            silent = (self.step % self.verbose_interval) != 0
            if not silent:
                density = pruning.report_density(model, silent=True)
                tqdm.tqdm.write(f"REPORTED DENSITY: {density}")
            pruning.apply_pruning_for_model(model)
            self.step += self.interval

    def on_train_end(self, logs=None):
        save_and_report_density(self, logs)

import tensorflow as tf
import tqdm

from modules.pruning import pruning_utils


class CosinePruningCallback(tf.keras.callbacks.Callback):
    def __init__(self, decay_steps, alpha, interval=100, verbose_interval=2000):
        super().__init__()
        self.schedule = tf.keras.experimental.CosineDecay(1.0, decay_steps, alpha)
        self.step = 0
        self.interval = interval
        self.verbose_interval = verbose_interval

    def on_train_batch_begin(self, batch, logs=None):
        if batch % self.interval == 0:
            self.step += self.interval
            density = self.schedule(self.step)

            model = pruning_utils.prune_l1(model=self.model,
                                           config={"sparsity": 1 - density},
                                           silent=True)
            density = pruning_utils.report_density(model, silent=True)
            silent = (self.step % self.verbose_interval) != 0
            if not silent:
                tqdm.tqdm.write(f"REPORTED DENSITY: {density}")
            pruning_utils.apply_pruning_for_model(model)


class PolynomialPruningCallback(tf.keras.callbacks.Callback):
    def __init__(self, decay_steps, alpha, interval=100, verbose_interval=2000):
        super().__init__()
        self.schedule = tf.keras.optimizers.schedules.PolynomialDecay(1.0,
                                                                      decay_steps,
                                                                      end_learning_rate=alpha,
                                                                      power=3.0)
        self.step = 0
        self.interval = interval
        self.verbose_interval = verbose_interval

    def on_train_batch_begin(self, batch, logs=None):
        if batch % self.interval == 0:
            self.step += self.interval
            density = self.schedule(self.step)

            silent = (self.step % self.verbose_interval) != 0
            model = pruning_utils.prune_l1(model=self.model,
                                           config={"sparsity": 1 - density},
                                           silent=silent)
            pruning_utils.apply_pruning_for_model(model)


class PiecewisePruningCallback(tf.keras.callbacks.Callback):
    def __init__(self, boundaries, values):
        super().__init__()
        assert len(boundaries) + 1 == len(values)
        self.schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries, values
        )
        self.value = values[0]
        self.step = 0

    def on_train_batch_begin(self, batch, logs=None):
        self.step += 1
        density = self.schedule(self.step)
        if density != self.value:
            self.value = density
            model = pruning_utils.prune_l1(model=self.model,
                                           config={"sparsity": 1 - density},
                                           silent=False)
            pruning_utils.apply_pruning_for_model(model)

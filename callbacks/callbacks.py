import tensorflow as tf
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

            silent = (self.step % self.verbose_interval) != 0
            model = pruning_utils.prune_l1(model=self.model,
                                           config={"sparsity": 1 - density},
                                           silent=silent)
            pruning_utils.apply_pruning_for_model(model)

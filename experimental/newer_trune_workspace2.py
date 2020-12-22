# %%

from experimental.toolkit import *
import tensorflow as tf
import numpy as np
from copy import deepcopy
from tools import models, datasets, utils
import tensorflow.keras.mixed_precision.experimental as mixed_precision

utils.set_memory_growth()
utils.set_precision(16)


def maybe_abs(mask):
    return mask


def mask_activation(mask):
    return tf.identity(mask)


def regularize(values):
    loss = 0
    for value in values:
        processed_value = maybe_abs(value)
        loss += tf.reduce_sum(processed_value) * regularizer_value
    return loss


class MaskedDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        self.kernel_mask = self.add_weight(
            name="kernel_mask",
            shape=self.kernel.shape,
            dtype=self.kernel.dtype,
            initializer="ones",
            trainable=False,
        )

    def call(self, x):
        multipler = mask_activation(self.kernel_mask)
        masked_w = tf.multiply(self.kernel, multipler)
        result = tf.matmul(x, masked_w)

        if self.use_bias:
            result = tf.add(result, self.bias)
        return self.activation(result)

    def set_pruning_mask(self, new_mask: np.ndarray):
        tf.assert_equal(new_mask.shape, self.kernel_mask.shape)
        self.kernel_mask.assign(new_mask)

    def apply_pruning_mask(self):
        self.kernel.assign(tf.multiply(self.kernel, self.kernel_mask))


class MaskedConv(tf.keras.layers.Conv2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        self.kernel_mask = self.add_weight(
            name="kernel_mask",
            shape=self.kernel.shape,
            dtype=self.kernel.dtype,
            initializer="ones",
            trainable=False,
        )

    def call(self, x):
        multipler = mask_activation(self.kernel_mask)
        masked_w = tf.multiply(self.kernel, multipler)
        result = tf.nn.conv2d(x, masked_w, strides=self.strides, padding=self.padding.upper())

        if self.use_bias:
            result = tf.add(result, self.bias)
        return self.activation(result)

    def set_pruning_mask(self, new_mask: np.ndarray):
        tf.assert_equal(new_mask.shape, self.kernel_mask.shape)
        self.kernel_mask.assign(new_mask)

    def apply_pruning_mask(self):
        self.kernel.assign(tf.multiply(self.kernel, self.kernel_mask))


checkpoint_lookup = {
    '2k': 'data/partial_training_checkpoints/VGG19_2000it/0.h5',
    '8k': 'data/partial_training_checkpoints/VGG19_8000it/0.h5',
    '16k': 'data/partial_training_checkpoints/VGG19_16000it/0.h5',
    '2k2': 'data/partial_training_checkpoints/VGG19_2000it/1.h5',
    '8k2': 'data/partial_training_checkpoints/VGG19_8000it/1.h5',
    'full_from_2k': 'data/VGG19_IMP03_ticket/130735/0.h5',
    'unrl_full1': 'data/VGG19_full_training/70754/0.h5',
    'unrl_full2': 'data/VGG19_full_training/70754/1.h5',
    'perf2': 'data/VGG19_IMP03_ticket/770423/10.h5',
}

config = utils.ddict(
    checkpoints=['8k']
)

net = models.VGG(input_shape=(32, 32, 3), n_classes=10, version=19,
                 DENSE_LAYER=MaskedDense, CONV_LAYER=MaskedConv)

perf_net = tf.keras.models.clone_model(net)
perf_net.load_weights(checkpoint_lookup['perf2'])
perf_kernel_masks = get_kernel_masks(perf_net)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = mixed_precision.LossScaleOptimizer(
    tf.keras.optimizers.SGD(learning_rate=100.0, momentum=0.99, nesterov=True),
    "dynamic")

kernel_masks = get_kernel_masks(net)
distributions = [tf.Variable(mask) for mask in kernel_masks]
nets = [net]

for i, ckp in enumerate(config.checkpoints):
    if len(nets) == i:
        nets.append(tf.keras.models.clone_model(net))
    nets[i].load_weights(checkpoint_lookup[ckp])
    nets[i].compile(deepcopy(optimizer), deepcopy(loss_fn))
    set_kernel_masks_object(nets[i], kernel_masks)

set_kernel_masks_values(distributions, 4.)
ds = datasets.cifar10(128, 128, shuffle=10000)

logger = create_logger(
    'full_loss',
    'train_loss',
    'valid_loss',
    'train_acc',
    'valid_acc',
    'max_gradient',
)

regularizer_value = tf.Variable(0.)


def set_kernel_masks_from_distributions(kernel_masks, distributions):
    for km, d in zip(kernel_masks, distributions):
        probs = tf.sigmoid(d)
        rnd = tf.random.uniform(shape=probs.shape, dtype=probs.dtype)
        km.assign(tf.cast(rnd <= probs, km.dtype))

set_kernel_masks_from_distributions(kernel_masks, distributions)

train_steps = []
for i in range(len(nets)):
    @tf.function
    def train_step(model, x, y):
        set_kernel_masks_from_distributions(kernel_masks, distributions)
        with tf.GradientTape() as tape:
            tape.watch(kernel_masks)
            tape.watch(distributions)
            outs = tf.cast(model(x), tf.float32)
            loss = model.loss(y, outs)
            logger['train_loss'](loss)

            loss += regularize(distributions)
            logger['full_loss'](loss)
            scaled_loss = model.optimizer.get_scaled_loss(loss)

        scaled_grads = tape.gradient(target=scaled_loss, sources=kernel_masks + distributions)
        grads = model.optimizer.get_unscaled_gradients(scaled_grads)

        max_gradient = tf.reduce_max([tf.reduce_max(tf.abs(grad)) for grad in grads])
        logger['train_acc'](tf.keras.metrics.sparse_categorical_accuracy(y, outs))
        logger['max_gradient'](max_gradient)

        if callable(optimizer.lr):
            grads = clip_many(grads, clip_at=0.1 / model.optimizer.lr(model.optimizer.iterations))
        else:
            grads = clip_many(grads, clip_at=0.1 / model.optimizer.lr)
        model.optimizer.apply_gradients(zip(grads, distributions + distributions))
        clip_many(kernel_masks, clip_at=10, inplace=True)


    train_steps.append(train_step)


@tf.function
def valid_step(model, x, y):
    outs = tf.cast(model(x), tf.float32)
    loss = model.loss(y, outs)
    logger['valid_loss'](loss)
    logger['valid_acc'](tf.keras.metrics.sparse_categorical_accuracy(y, outs))


@tf.function
def train_epoch(models, steps):
    for x, y in ds['train'].take(steps):
        for i, model in enumerate(models):
            train_steps[i](model, x, y)
        tf.numpy_function(update_pbar, inp=[], Tout=[])


@tf.function
def valid_epoch(model):
    for x, y in ds['valid']:
        valid_step(model, x, y)


def update_pbar():
    pbar.add(1, values=[
        ('full_loss', logger['full_loss'].result()),
        ('train_loss', logger['train_loss'].result()),
        ('train_acc', logger['train_acc'].result()),
        ('max_gradient', logger['max_gradient'].result()),
    ])


valid_epoch(net)
get_logger_results(logger, show=False)

# %%

EPOCHS = 4
STEPS = 2000

regularizer_schedule = {
    0: 1e-7,
    # 4: 2e-7,
    # 6: 3e-7,
    # 7: 4e-7,
    # 9: 5e-7,
    # 10: 6e-7,
    # 11: 7e-7,
    # 12: 8e-7,
    # 13: 9e-7,
    # 14: 1e-6,
}

pbar = tf.keras.utils.Progbar(target=EPOCHS * STEPS, width=10, interval=1.0,
                              stateful_metrics=[
                                  'train_loss',
                                  'train_acc',
                                  'max_gradient'
                              ])

for epoch in range(EPOCHS):
    if epoch in regularizer_schedule:
        regularizer_value.assign(regularizer_schedule[epoch])

    train_epoch(nets, STEPS)
    for net in nets:
        valid_epoch(net)

    mask = update_mask_info(kernel_masks, mask_activation, logger)
    f1, prc, rec, thr, density = compare_masks(perf_kernel_masks, kernel_masks,
                                               mask_activation=mask_activation)
    logger['f1'] = f1
    logger['density2'] = density

    print('\r', end='')
    logs = get_logger_results(logger, show=True)
    visualize_masks(kernel_masks, mask_activation)

# %%

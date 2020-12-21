# %%

import time
import tensorflow as tf
import numpy as np
from copy import deepcopy
from tools import models, datasets, utils
import tensorflow.keras.mixed_precision.experimental as mixed_precision
import matplotlib.pyplot as plt

utils.set_memory_growth()
# utils.set_precision(16)


# %%

def get_differentiable_kernel_multipler(mask):
    mask = tf.tanh(mask)
    return mask


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
        multipler = get_differentiable_kernel_multipler(self.kernel_mask)
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
        multipler = get_differentiable_kernel_multipler(self.kernel_mask)
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


def get_kernel_masks(model):
    return [w for w in model.weights if 'kernel_mask' in w.name]


def get_kernels(model):
    return [l.kernel for l in model.layers if hasattr(l, 'kernel')]


def set_kernel_masks_value(model, masks):
    for i, kernel in enumerate(get_kernel_masks(model)):
        if isinstance(masks, int) or isinstance(masks, float):
            mask = np.ones_like(kernel.numpy()) * masks
        else:
            mask = masks[i]
        kernel.assign(mask)


def set_kernel_masks_object(model, masks):
    layers = (l for l in model.layers if hasattr(l, 'kernel_mask'))
    for l, km in zip(layers, masks):
        l.kernel_mask = km


def clip_many(values, value, inplace=False):
    if inplace:
        for v in values:
            v.assign(tf.clip_by_value(v, -value, value))
    else:
        r = []
        for v in values:
            r.append(tf.clip_by_value(v, -value, value))
        return r


def visualize_masks(masks):
    plt.figure(figsize=(7, 8))
    c = np.concatenate([mask.numpy().flatten() for mask in masks])
    plt.subplot(2, 1, 1)
    plt.hist(c, bins=30)

    c = np.concatenate([mask.numpy().flatten() for mask in masks[:4]])
    plt.subplot(2, 1, 2)
    plt.hist(c, bins=30)
    plt.show()


checkpoint_lookup = {
    '2k': 'data/partial_training_checkpoints/VGG19_2000it/0.h5',
    '8k': 'data/partial_training_checkpoints/VGG19_8000it/0.h5',
    '16k': 'data/partial_training_checkpoints/VGG19_16000it/0.h5',
    '2k2': 'data/partial_training_checkpoints/VGG19_2000it/1.h5',
    '8k2': 'data/partial_training_checkpoints/VGG19_8000it/1.h5',
    'full_from_2k': 'data/VGG19_IMP03_ticket/130735/0.h5',
    'unrl_full1': 'data/VGG19_full_training/70754/0.h5',
    'unrl_full2': 'data/VGG19_full_training/70754/1.h5',
}

config = utils.ddict(
    checkpoints=['8k']
)

net = models.VGG(input_shape=(32, 32, 3), n_classes=10, version=19,
                 DENSE_LAYER=MaskedDense, CONV_LAYER=MaskedConv)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = mixed_precision.LossScaleOptimizer(
    tf.keras.optimizers.SGD(learning_rate=100.0, momentum=0.99, nesterov=True),
    "dynamic")

net.compile(optimizer, loss_fn)

kernel_masks = get_kernel_masks(net)
nets = [net]

for i, ckp in enumerate(config.checkpoints):
    if len(nets) == i:
        nets.append(tf.keras.models.clone_model(net))
    nets[i].load_weights(checkpoint_lookup[ckp])
    nets[i].compile(deepcopy(optimizer), deepcopy(loss_fn))
    set_kernel_masks_object(nets[i], kernel_masks)

set_kernel_masks_value(net, 4.)
ds = datasets.cifar10(128, 128)


def create_logger(*keys):
    return {key: tf.keras.metrics.Mean() for key in keys}


def get_logger_results(logger, show=True):
    results = {}
    for key, value in logger.items():
        if hasattr(value, 'result'):
            results[key] = value.result().numpy()
            value.reset_states()
        else:
            results[key] = value
    if show:
        print(*[f"{key[:2]}-{key[-2:]}: {str(value)[:6]:<6}"
                for key, value in results.items()], sep=' | ')
    return results


logger = create_logger(
    'full_loss',
    'train_loss',
    'valid_loss',
    'train_acc',
    'valid_acc',
    'max_gradient',
)

regularizer_value = tf.Variable(0.)


def regularize(values):
    loss = 0
    for value in values:
        processed_value = value
        loss += tf.reduce_sum(tf.abs(processed_value)) * regularizer_value
    return loss


train_steps = []
for i in range(len(nets)):
    @tf.function
    def train_step(model, x, y):
        with tf.GradientTape() as tape:
            tape.watch(kernel_masks)
            outs = tf.cast(model(x), tf.float32)
            loss = model.loss(y, outs)
            logger['train_loss'](loss)

            loss += regularize(kernel_masks)
            logger['full_loss'](loss)
            scaled_loss = model.optimizer.get_scaled_loss(loss)

        scaled_grads = tape.gradient(target=scaled_loss, sources=kernel_masks)
        grads = model.optimizer.get_unscaled_gradients(scaled_grads)

        max_gradient = tf.reduce_max([tf.reduce_max(tf.abs(grad)) for grad in grads])
        logger['train_acc'](tf.keras.metrics.sparse_categorical_accuracy(y, outs))
        logger['max_gradient'](max_gradient)

        grads = clip_many(grads, value=0.01 / model.optimizer.learning_rate)
        model.optimizer.apply_gradients(zip(grads, kernel_masks))
        clip_many(kernel_masks, value=10, inplace=True)


    train_steps.append(train_step)


@tf.function
def valid_step(model, x, y):
    outs = tf.cast(model(x), tf.float32)
    loss = model.loss(y, outs)
    logger['valid_loss'](loss)
    logger['valid_acc'](tf.keras.metrics.sparse_categorical_accuracy(y, outs))


def update_pbar():
    pbar.add(1, values=[
        ('train_loss', logger['train_loss'].result()),
        ('train_acc', logger['train_acc'].result()),
        ('max_gradient', logger['max_gradient'].result()),
    ])


@tf.function
def train_epoch(models, steps):
    for x, y in ds['train'].take(steps):
        for i, model in enumerate(models):
            train_steps[i](model, x, y)
        tf.numpy_function(update_pbar, [], Tout=[])


@tf.function
def valid_epoch(model):
    for x, y in ds['valid']:
        valid_step(model, x, y)


def get_density(kernel_masks):
    mask = np.concatenate([
        tf.abs(get_differentiable_kernel_multipler(mask)).numpy().flatten()
        for mask in kernel_masks
    ])
    return np.mean(mask)


# %%

EPOCHS = 8
STEPS = 2000

regularizer_value.assign(1e-7)
pbar = tf.keras.utils.Progbar(target=EPOCHS * STEPS, width=10,
                              stateful_metrics=[
                                  'train_loss',
                                  'train_acc',
                                  'max_gradient'
                              ])

for epoch in range(EPOCHS):
    train_epoch(nets, STEPS)
    for net in nets:
        valid_epoch(net)

    logger['density'] = get_density(kernel_masks)

    print('\r', end='')
    logs = get_logger_results(logger, show=True)
    visualize_masks(kernel_masks)

# %%

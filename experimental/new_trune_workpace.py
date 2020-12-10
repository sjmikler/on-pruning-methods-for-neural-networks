import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tools import datasets, models, pruning, utils
import tensorflow.keras.mixed_precision.experimental as mixed_precision
import tensorflow_addons as tfa

utils.set_memory_growth()
utils.set_precision(16)


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

        self.sparsity = 1 - np.mean(self.kernel_mask.numpy())

    def call(self, x):
        mask = tf.sigmoid(self.kernel_mask)
        masked_w = tf.multiply(self.kernel, mask)
        # masked_w = masked_w / tf.reduce_mean(self.kernel_mask)

        result = tf.matmul(x, masked_w)

        if self.use_bias:
            result = tf.add(result, self.bias)

        return self.activation(result)

    def set_pruning_mask(self, new_mask: np.ndarray):
        """
        :param new_mask: mask of the same shape as `layer.kernel`
        :return: None
        """
        tf.assert_equal(new_mask.shape, self.kernel_mask.shape)
        self.kernel_mask.assign(new_mask)
        self.sparsity = 1 - np.mean(self.kernel_mask.numpy())
        self.left_unpruned = np.sum(self.kernel_mask.numpy() == 1)

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
        self.sparsity = 1 - np.mean(self.kernel_mask.numpy())

    def call(self, x):
        mask = tf.sigmoid(self.kernel_mask)
        masked_w = tf.multiply(self.kernel, mask)
        # masked_w = masked_w / tf.reduce_mean(self.kernel_mask)

        result = tf.nn.conv2d(
            x, masked_w, strides=self.strides, padding=self.padding.upper()
        )

        if self.use_bias:
            result = tf.add(result, self.bias)

        return self.activation(result)

    def set_pruning_mask(self, new_mask: np.ndarray):
        """
        :param new_mask: mask of the same shape as `layer.kernel`
        :return: None
        """
        tf.assert_equal(new_mask.shape, self.kernel_mask.shape)
        self.kernel_mask.assign(new_mask)
        self.sparsity = 1 - np.mean(self.kernel_mask.numpy())
        self.left_unpruned = np.sum(self.kernel_mask.numpy() == 1)

    def apply_pruning_mask(self):
        self.kernel.assign(tf.multiply(self.kernel, self.kernel_mask))


# %%

ds = datasets.cifar10(128, 128, shuffle=20000)
# ds['train'] = ds['train'].map(
#     lambda x, y: (tfa.image.random_cutout(x, mask_size=6, constant_values=0), y))

# schedule = tf.optimizers.schedules.PiecewiseConstantDecay([8000, 12000], [100, 10, 1])
optimizer = tf.optimizers.SGD(learning_rate=100, momentum=0.99, nesterov=True)
optimizer = mixed_precision.LossScaleOptimizer(optimizer, "dynamic")
loss_fn = tf.losses.SparseCategoricalCrossentropy(True)

model = models.VGG((32, 32, 3), n_classes=10, version=19,
                   CONV_LAYER=MaskedConv, DENSE_LAYER=MaskedDense)
model.load_weights('data/partial_training_checkpoints/VGG19_init_8000_v2.h5')

full_loss_metric = tf.metrics.Mean()
loss_metric = tf.metrics.SparseCategoricalCrossentropy(True)
accu_metric = tf.metrics.SparseCategoricalAccuracy(True)
kernel_masks = [w for w in model.weights if "kernel_mask" in w.name]

all_models = [model]
paths = [
    'data/partial_training_checkpoints/VGG19_init_8000v2_2/535079/0.h5'
]
for path in paths:
    m = models.VGG((32, 32, 3), n_classes=10, version=19,
                   CONV_LAYER=MaskedConv, DENSE_LAYER=MaskedDense)
    m.load_weights(path)

    idx = 0
    for layer in m.layers:
        if hasattr(layer, 'kernel_mask'):
            layer.kernel_mask = kernel_masks[idx]
            idx += 1
    all_models.append(m)

for kernel in kernel_masks:
    kernel.assign(np.ones_like(kernel.numpy()) * 3)

decay = tf.Variable(1e-6)


def get_and_reset(metric):
    x = metric.result()
    metric.reset_states()
    return x


def reg_fn():
    loss = 0
    for mask in kernel_masks:
        loss += tf.reduce_sum(tf.sigmoid(mask))
    return loss


@tf.function
def train_step(model, x, y):
    with tf.GradientTape() as tape:
        tape.watch(kernel_masks)
        outs = model(x, training=True)
        outs = tf.cast(outs, tf.float32)
        loss = loss_fn(y, outs)
        loss += reg_fn() * decay
        scaled_loss = optimizer.get_scaled_loss(loss)

    loss_metric(y, outs)
    accu_metric(y, outs)
    full_loss_metric(loss)
    grads = tape.gradient(scaled_loss, kernel_masks)
    grads = optimizer.get_unscaled_gradients(grads)
    optimizer.apply_gradients(zip(grads, kernel_masks))

    for mask in kernel_masks:
        mask.assign(tf.clip_by_value(mask, -15, 15))


@tf.function
def valid_step(model, x, y):
    outs = model(x, training=False)
    outs = tf.cast(outs, tf.float32)
    loss_metric(y, outs)
    accu_metric(y, outs)


@tf.function
def valid_epoch(model, ds):
    for x, y in ds:
        valid_step(model, x, y)


def report_density(model, detailed=False, sigmoid=False):
    nonzero = 0
    max_nonzero = 0

    more_than_half = 0
    for layer in model.layers:
        if hasattr(layer, 'kernel_mask'):
            km = layer.kernel_mask.numpy()
            more_than_half += np.sum(km > 0)

            if sigmoid:
                km = tf.sigmoid(km).numpy()

            max_nonzero += km.size
            nonzero_here = km.sum()

            if detailed:
                print(f"density of {layer.name:>16}: {np.mean(km):6.4f}")
            nonzero += nonzero_here

    print(f"Real density: {more_than_half / max_nonzero:6.4f}")
    return nonzero / max_nonzero


for m in all_models:
    valid_epoch(m, ds['valid'])
    print(f"V LOSS: {get_and_reset(loss_metric):6.3f}",
          f"V ACCU: {get_and_reset(accu_metric):6.4f}",
          sep=' | ')

plt.hist(np.concatenate([km.numpy().flatten() for km in kernel_masks[:3]]), bins=40)
plt.show()

# %%

decay.assign(5e-7)

NUM_ITER = 8000
REP_ITER = 200
VAL_ITER = 2000

t0 = time.time()

for step, (x, y) in enumerate(ds['train']):
    for m in all_models:
        train_step(m, x, y)

    if (step + 1) % REP_ITER == 0:
        print(
            f"STEP: {step + 1:^8}",
            f"T FULL: {get_and_reset(full_loss_metric):8.3f}",
            f"T LOSS: {get_and_reset(loss_metric):6.3f}",
            f"T ACCU: {get_and_reset(accu_metric):6.4f}",
            f"DENSITY: {report_density(m, sigmoid=True):8.6f}",
            f"TIME: {time.time() - t0:6.0f}",
            sep=' | ')

    if (step + 1) % VAL_ITER == 0:
        for m in all_models:
            valid_epoch(m, ds['valid'])

        print(
            f"{'VALIDATION':^14}",
            f"V LOSS: {get_and_reset(loss_metric):6.3f}",
            f"V ACCU: {get_and_reset(accu_metric):6.4f}",
            f"DENSITY: {report_density(m, sigmoid=True):8.6f}",
            f"TIME: {time.time() - t0:6.0f}",
            sep=' | ')

        report_density(m, detailed=True, sigmoid=True)
        plt.hist(np.concatenate([km.numpy().flatten() for km in kernel_masks[:3]]), bins=40)
        plt.show()

        m.save_weights(f'temp/new_trune_workspace_ckp.h5', save_format="h5")

    if (step + 1) % NUM_ITER == 0:
        break

# %%

model2 = models.VGG((32, 32, 3), n_classes=10, version=19,
                    CONV_LAYER=MaskedConv, DENSE_LAYER=MaskedDense)
model2.load_weights('temp/new_trune_workspace_ckp.h5')
km = [w for w in model2.weights if 'kernel_mask' in w.name]

print(report_density(model2, sigmoid=True))
for m in km:
    mn = m.numpy()
    mn[mn < 0] = 0
    mn[mn > 0] = 1
    m.assign(mn)

print(report_density(model2))
# model2.save_weights('temp/new_trune_workspace_ckp4.h5')

# %%

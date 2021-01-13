import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from experimental import toolkit
from tools import models, datasets, utils

utils.set_memory_growth()
utils.set_precision(16)


def mask_activation(mask):
    return tf.sigmoid(mask)


conv, dense = toolkit.create_layers(mask_activation)
model = models.VGG((32, 32, 3), n_classes=10, version=19,
                   CONV_LAYER=conv,
                   DENSE_LAYER=dense)

ckp_lookup = {
    '2k': 'data/partial_training_checkpoints/VGG19_2000it/0.h5',
    '8k': 'data/partial_training_checkpoints/VGG19_8000it/0.h5'
}

model.load_weights(ckp_lookup['8k'])
ds = datasets.cifar10()
reg_rate = tf.Variable(1e-7)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=1.,
                                    momentum=0.99,
                                    nesterov=True)
optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale=4096)

kernel_masks = toolkit.get_kernel_masks(model)
toolkit.set_kernel_masks_values(kernel_masks, 5.)
logger = toolkit.Logger(column_width=10)

kernels = toolkit.get_kernels(model)
org_kernels = [k.numpy() for k in kernels]


def mask_regularization(masks):
    l = 0.
    for km in masks:
        l += tf.reduce_sum(km + 15) * reg_rate
    return l


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        tape.watch(kernel_masks)
        outs = model(x, training=True)
        outs = tf.cast(outs, tf.float32)
        loss = loss_fn(y, outs)
        logger['train_loss'](loss)

        reg_loss = mask_regularization(kernel_masks)
        logger['regularization_loss'](reg_loss)

        loss = loss + reg_loss
        scaled_loss = optimizer.get_scaled_loss(loss)
    scaled_grads = tape.gradient(scaled_loss, kernel_masks)
    grads = optimizer.get_unscaled_gradients(scaled_grads)

    for i, kernel in enumerate(kernels):
        grads[i] = grads[i] / tf.abs(kernel)

    optimizer.apply_gradients(zip(grads, kernel_masks))
    toolkit.clip_many(kernel_masks, 15, inplace=True)

    logger['train_acc'](tf.keras.metrics.sparse_categorical_accuracy(y, outs))
    # return grads


@tf.function
def valid_step(x, y):
    outs = model(x, training=False)
    outs = tf.cast(outs, tf.float32)
    loss = loss_fn(y, outs)
    logger['valid_loss'](loss)
    logger['valid_acc'](tf.keras.metrics.sparse_categorical_accuracy(y, outs))


for x, y in ds['valid']:
    valid_step(x, y)
logger.show()

# %%

EPOCHS = 8
ITERS = 2000
reg_rate.assign(1e-6)

all_grads = []
logger.show_header()

for epoch in range(EPOCHS):
    t0 = time.time()
    for x, y in ds['train'].take(ITERS):
        # if np.random.rand() < 0.1:
        #     for k, ok in zip(kernels, org_kernels):
        #         nk = ok * (1 + (np.random.rand(*ok.shape) - 0.5))
        #         k.assign(nk)

        grads = train_step(x, y)
        # if grads:
        #     all_grads.append([g.numpy() for g in grads])
    epoch_time = time.time() - t0

    for x, y in ds['valid']:
        valid_step(x, y)

    toolkit.update_mask_info(kernel_masks, mask_activation, logger)
    toolkit.visualize_masks(kernel_masks, mask_activation)
    logger['epoch_time'] = epoch_time
    logger.show()

toolkit.prune_and_save_model(model, mask_activation, threshold=0.01,
                             path='temp/new_trune_workspace_ckp.h5')

# %%

means = [[] for _ in range(len(all_grads[0]))]

for grads in all_grads:
    grads = [np.abs(g) for g in grads]
    grads = [np.reshape(g, -1) for g in grads]
    for i, g in enumerate(grads):
        means[i].append(np.mean(g))

# %%

plt.figure(figsize=(5, 15))

for i, m in enumerate(means):
    plt.plot(range(len(m)), m, label=f'layer {i}')

# plt.ylim(0, 0.25 * 1e-5)
plt.legend()
plt.show()

# %%

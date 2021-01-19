import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tqdm import tqdm

from experimental import toolkit
import tools.datasets as datasets
import tools.models as models
import tools.utils as utils
import tools.pruning as pruning

utils.set_memory_growth()
utils.set_precision(16)


def mask_activation(mask):
    return tf.sigmoid(mask)


model = models.VGG((32, 32, 3), n_classes=10, version=19, l2_reg=1e-4)
# model.load_weights('data/VGG19_IMP03_ticket/770423/10.h5')
ds = datasets.cifar10()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[32000, 48000, 64000], values=[0.1, 0.02, 0.004, 0.0008]
)
optimizer = tf.keras.optimizers.SGD(learning_rate=schedule,
                                    momentum=0.9,
                                    nesterov=True)
optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale=4096)

# %%

logger = toolkit.Logger(column_width=10)


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        outs = model(x, training=True)
        outs = tf.cast(outs, tf.float32)
        loss = loss_fn(y, outs)
        logger['train_loss'](loss)
        loss += tf.add_n(model.losses)

        scaled_loss = optimizer.get_scaled_loss(loss)
    scaled_grads = tape.gradient(scaled_loss, model.trainable_weights)
    grads = optimizer.get_unscaled_gradients(scaled_grads)

    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    logger['train_acc'](tf.keras.metrics.sparse_categorical_accuracy(y, outs))
    return grads


@tf.function
def valid_step(x, y):
    outs = model(x, training=False)
    outs = tf.cast(outs, tf.float32)
    loss = loss_fn(y, outs)
    logger['valid_loss'](loss)
    logger['valid_acc'](tf.keras.metrics.sparse_categorical_accuracy(y, outs))


# %%

for epoch in range(80):
    t0 = time.time()
    for x, y in ds['train'].take(2000):
        grads = train_step(x, y)
    logger['time_train'] = time.time() - t0

    t0 = time.time()
    for x, y in ds['valid']:
        valid_step(x, y)
    logger['time_valid'] = time.time() - t0
    logger.show()

# %%

toolkit.visualize_masks(toolkit.get_kernel_masks(model),
                        mask_activation=lambda x: x)

nm = models.VGG((32, 32, 3), n_classes=10, version=19)
nkm = [1 - m for m in kernel_masks]
toolkit.set_kernel_masks_values_on_model(nm, nkm)
toolkit.visualize_masks(nkm, mask_activation=lambda x: x)
nm.save_weights('temp/ensemble_model_test.h5')

# %%

preds = model.predict(ds['valid'])
target = tf.concat([y for x, y in ds['valid']], 0)

for cp in tqdm([
    'data/VGG19_IMP03_ticket/775908/9.h5',
    'data/VGG19_IMP03_ticket/775908/8.h5',
    'data/VGG19_IMP03_ticket/775908/7.h5',
    'data/VGG19_IMP03_ticket/775908/6.h5',
    'data/VGG19_IMP03_ticket/775908/5.h5',
    'data/VGG19_IMP03_ticket/775908/4.h5',
    'data/VGG19_IMP03_ticket/775908/3.h5',
    'data/VGG19_IMP03_ticket/775908/2.h5',
    'data/VGG19_IMP03_ticket/775908/1.h5',
    'data/VGG19_IMP03_ticket/775908/0.h5',
]):
    m = models.VGG((32, 32, 3), n_classes=10, version=19)
    m.load_weights(cp)
    p = m.predict(ds['valid'])
    preds += p

tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(target, preds))

# %%

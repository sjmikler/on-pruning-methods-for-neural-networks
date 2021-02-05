# %%

import tensorflow as tf
import numpy as np
import time

from tools import *

utils.set_memory_growth()
utils.set_precision(16)

model = models.VGG((32, 32, 3), n_classes=10, version=19, l2_reg=1e-4)
model.load_weights('data/VGG19_IMP03_ticket/770423/10.h5')
masks = toolkit.get_kernel_masks(model)
kernels = toolkit.get_kernels(model)
toolkit.visualize_masks(masks)


# %%

def numpy_all(arrays):
    return [a if isinstance(a, np.ndarray) else a.numpy() for a in arrays]


def oversample_dense_to_structured(mask):
    mask = mask.T
    new_mask = np.zeros_like(mask)
    for i, neuron in enumerate(mask):
        if any(neuron):
            new_mask[i] = 1
    new_mask = new_mask.T
    toolkit.concatenate_flattened(new_mask)
    print(f'Oversampled mask density: {np.mean(new_mask)}')
    return new_mask


def oversample_conv_to_structured(mask):
    shape = mask.shape
    mask = mask.reshape(-1, shape[-1])
    new_mask = oversample_dense_to_structured(mask)
    new_mask = new_mask.reshape(shape)
    return new_mask


def oversample_masks_to_structured(masks):
    new_masks = []
    for mask in masks:
        if len(mask.shape) == 2:
            new_masks.append(oversample_dense_to_structured(mask))
        elif len(mask.shape) == 4:
            new_masks.append(oversample_conv_to_structured(mask))
        else:
            raise Exception('Mask must be either for conv or dense layers!')
    return new_masks


strucuted_masks = oversample_masks_to_structured(numpy_all(masks))
toolkit.set_kernel_masks_values(masks, strucuted_masks)

inv_mask = [m.numpy().copy() for m in masks]
interesting_kernels = []
for idx in [9, 10, 11, 12, 13, 14, 15]:
    inv_mask[idx] = 1 - masks[idx]
    interesting_kernels.append(kernels[idx])
interesting_kernels_id = [id(k) for k in interesting_kernels]
kernels_id = [id(k) for k in kernels]
toolkit.visualize_masks(masks)

# %%

m = tf.keras.models.clone_model(model)
toolkit.set_kernel_masks_values_on_model(m, inv_mask)
toolkit.visualize_masks(toolkit.get_kernel_masks(m))

trainable_kernels = []
trainable_others = []
for w1, w2 in zip(model.weights, m.weights):
    if id(w1) in interesting_kernels_id:
        trainable_kernels.append(w2)
    else:
        w2.assign(w1)
        if id(w1) not in kernels_id and w2.trainable:
            trainable_others.append(w2)

# %%

logger = toolkit.Logger(column_width=10)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[10000, 14000, 18000],
    values=[0.1, 0.02, 0.004, 0.0008]
)
optimizer = tf.keras.optimizers.SGD(learning_rate=schedule,
                                    momentum=0.9,
                                    nesterov=True)
optimizer = utils.mixed_precision_optimizer(optimizer, loss_scale=4096)
ds = datasets.cifar10(128, 256)

trainable_all = trainable_kernels + trainable_others


@tf.function
def train_step(model, x, y):
    with tf.GradientTape() as tape:
        outs = model(x, training=True)
        outs = tf.cast(outs, tf.float32)
        loss = loss_fn(y, outs)
        logger['training_loss'](loss)

        regloss = tf.add_n(model.losses)
        logger['regularization_loss'](regloss)

        loss = loss + regloss
        scaled_loss = optimizer.get_scaled_loss(loss)
    scaled_grads = tape.gradient(scaled_loss, trainable_all)
    unscaled_grads = optimizer.get_unscaled_gradients(scaled_grads)
    optimizer.apply_gradients(zip(unscaled_grads, trainable_all))

    acc = tf.keras.metrics.sparse_categorical_accuracy(y, outs)
    logger['training_acc'](tf.reduce_mean(acc))


@tf.function
def valid_step(model, x, y):
    outs = model(x, training=False)
    outs = tf.cast(outs, tf.float32)
    loss = loss_fn(y, outs)
    logger['validation_loss'](loss)

    acc = tf.keras.metrics.sparse_categorical_accuracy(y, outs)
    logger['validation_acc'](tf.reduce_mean(acc))


# %%

iters = 22000
steps_per_epoch = 2000

for epoch in range(iters // steps_per_epoch):
    t0 = time.time()
    for x, y in ds['train'].take(steps_per_epoch):
        train_step(m, x, y)
    logger['training_time'] = time.time() - t0

    for x, y in ds['valid']:
        valid_step(m, x, y)
    logger.show()

# %%

p1 = model.predict(ds['valid'])
p2 = m.predict(ds['valid'])
target = tf.concat([y for x, y in ds['valid']], 0)

psum = p1 + p2
print(tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(target, p1)))
print(tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(target, p2)))
print(tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(target, psum)))

# %%

# %%

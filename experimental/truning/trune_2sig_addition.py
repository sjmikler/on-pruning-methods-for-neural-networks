# %%

import time
from tools.pruning_toolkit import *
import tensorflow as tf
import numpy as np
from copy import deepcopy

from tools import models, datasets, utils
import tensorflow.keras.mixed_precision.experimental as mixed_precision
import tqdm

utils.set_memory_growth()
utils.set_precision(16)


def maybe_abs(mask):
    return tf.identity(mask)


def mask_activation(mask):
    return tf.sigmoid(mask[0]) - tf.sigmoid(mask[1])


MaskedConv, MaskedDense = create_masked_layers(mask_activation)
MaskedConv_vsign, MaskedDense_vsign = create_layers_vsign(mask_activation)


def regularize(values):
    loss = 0
    for value in values:
        processed_value = maybe_abs(value) + 10
        loss += tf.reduce_sum(processed_value) * regularizer_value
    return loss


mask_initial_value = 4.
optimizer = mixed_precision.LossScaleOptimizer(
    tf.keras.optimizers.SGD(learning_rate=100.0, momentum=0.9, nesterov=True),
    loss_scale=4096)

checkpoint_lookup = {
    '2k': 'data/partial_training_checkpoints/VGG19_2000it/0.h5',
    '8k': 'data/partial_training_checkpoints/VGG19_8000it/0.h5',
    '16k': 'data/partial_training_checkpoints/VGG19_16000it/0.h5',
    '2k2': 'data/partial_training_checkpoints/VGG19_2000it/1.h5',
    '8k2': 'data/partial_training_checkpoints/VGG19_8000it/1.h5',
    'full_from_2k': 'data/VGG19_IMP03_ticket/130735/0.h5',
    'unrl_full1': 'data/VGG19_full_training/70754/0.h5',
    'unrl_full2': 'data/VGG19_full_training/70754/1.h5',
    'perf': 'data/VGG19_IMP03_ticket/770423/10.h5',
    'perf2': 'data/VGG19_IMP03_ticket/775908/10.h5',
}

choosen_checkpoints = ['8k']

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

temp_net = models.VGG(input_shape=(32, 32, 3), n_classes=10, version=19,
                      DENSE_LAYER=MaskedDense, CONV_LAYER=MaskedConv)
temp_net.load_weights(checkpoint_lookup[choosen_checkpoints[0]])
temp_net.compile(deepcopy(optimizer), deepcopy(loss_fn))

net = models.VGG(input_shape=(32, 32, 3), n_classes=10, version=19,
                 DENSE_LAYER=MaskedDense_vsign, CONV_LAYER=MaskedConv_vsign)
set_all_weights_from_model(net, temp_net)

perf_net = tf.keras.models.clone_model(temp_net)
perf_net.load_weights(checkpoint_lookup['perf'])
perf_kernel_masks = get_kernel_masks(perf_net)

kernel_masks = get_kernel_masks(net)
regularizer_value = tf.Variable(0.)

################ IF MASK SAMPLING
mask_distributions = kernel_masks
all_differentiable = kernel_masks
all_updatable = kernel_masks

# set_kernel_masks_values(mask_distributions, mask_initial_value)
for km in mask_distributions:
    ones = np.ones_like(km.numpy())
    ones[0] = mask_initial_value
    ones[1] = -mask_initial_value
    km.assign(ones)

net.compile(deepcopy(optimizer), deepcopy(loss_fn))
ds = datasets.cifar10(128, 128, shuffle_train=10000)

logger = Logger(column_width=10)


@tf.function
def train_step(model, x, y):
    with tf.GradientTape() as tape:
        tape.watch(all_differentiable)
        outs = model(x, training=True)
        outs = tf.cast(outs, tf.float32)
        loss = model.loss(y, outs)
        logger['train_loss'](loss)

        loss += regularize(mask_distributions)
        logger['full_loss'](loss)
        scaled_loss = model.optimizer.get_scaled_loss(loss)

    scaled_grads = tape.gradient(target=scaled_loss, sources=all_differentiable)
    grads = model.optimizer.get_unscaled_gradients(scaled_grads)

    max_gradient = tf.reduce_max([tf.reduce_max(tf.abs(grad)) for grad in grads])
    logger['train_acc'](tf.keras.metrics.sparse_categorical_accuracy(y, outs))
    logger['max_gradient'](max_gradient)

    if callable(model.optimizer.lr):
        grads = clip_many(grads, clip_at=0.1 / model.optimizer.lr(model.optimizer.iterations))
    else:
        grads = clip_many(grads, clip_at=0.1 / model.optimizer.lr)
    model.optimizer.apply_gradients(zip(grads, all_updatable))
    clip_many(mask_distributions, clip_at=10, inplace=True)


@tf.function
def valid_step(model, x, y):
    outs = model(x, training=False)
    outs = tf.cast(outs, tf.float32)
    loss = model.loss(y, outs)
    logger['valid_loss'](loss)
    logger['valid_acc'](tf.keras.metrics.sparse_categorical_accuracy(y, outs))


@tf.function
def train_epoch(model, steps):
    for x, y in ds['train'].take(steps):
        train_step(model, x, y)
        tf.numpy_function(update_pbar, inp=[], Tout=[])


@tf.function
def valid_epoch(model, ds):
    for x, y in ds:
        valid_step(model, x, y)


def update_pbar():
    pbar.update(1)
    pbar.set_postfix(logger.peek('full_loss', 'train_loss',
                                 'train_acc', 'max_gradient'), refresh=False)


valid_epoch(net, ds['train'].take(50))
mask = log_mask_info(mask_distributions, mask_activation, logger)

f1, prc, rec, thr, density = compare_masks(perf_kernel_masks,
                                           mask_distributions,
                                           mask_activation=mask_activation)
logger['f1_to_perf'] = f1
logger['rec_to_perf'] = rec
logger['thr_to_perf'] = thr
logger['f1_density'] = density
logger.show()

visualize_masks(mask_distributions, mask_activation)

# %%

EPOCHS = 4
STEPS = 2000

regularizer_schedule = {
    0: 1e-6,
}

logger.show_header()
pbar = tqdm.tqdm(total=EPOCHS * STEPS, position=0, mininterval=0.5)

for epoch in range(EPOCHS):
    if epoch in regularizer_schedule:
        regularizer_value.assign(regularizer_schedule[epoch])

    t0 = time.time()
    train_epoch(net, STEPS)
    valid_epoch(net, ds['valid'])
    logger['epoch_time'] = time.time() - t0

    mask = log_mask_info(mask_distributions, mask_activation, logger)
    f1, prc, rec, thr, density = compare_masks(
        perf_kernel_masks,
        mask_distributions,
        mask_activation=mask_activation,
        # force_sparsity=0.98
    )
    logger['f1_to_perf'] = f1
    logger['rec_to_perf'] = rec
    logger['thr_to_perf'] = thr
    logger['f1_density'] = density

    print('\r', end='')
    logger.show()
    visualize_masks(mask_distributions, mask_activation)

pbar.close()
# prune_and_save_model(net, mask_activation, threshold=0.01,
#                      path='temp/new_trune_workspace_ckp.h5')

# %%

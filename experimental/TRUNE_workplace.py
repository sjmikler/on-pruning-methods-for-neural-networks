# %%

from IPython import embed
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow.keras.mixed_precision.experimental as mixed_precision
from tqdm import tqdm
import time

from sklearn.utils import shuffle
from itertools import islice
from tools.utils import set_memory_growth

from tools import datasets, layers, models
from tools.pruning import apply_pruning_for_model, get_pruning_mask, report_density
import tensorflow_addons as tfa

set_memory_growth()
policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_policy(policy)

sns.set()

dataset = datasets.cifar10()
model = models.VGG((32, 32, 3), n_classes=10, version=19)
# "logs/LTR444_shuffling_variants/start/78125/start_for_trune800.h5" 800 it
# model.load_weights("temp/11.20_trune_gumbel_things/VGG19truneTRY4/63446/start_for_trune800.h5")
model.load_weights(
    'data/start_for_trune800.h5')  # 8000 it
# model.load_weights('temp/11.20_trune_gumbel_things/VGG19truneTRY4/86664/start_for_trune800.h5')  # FULL
# model.load_weights('temp/11.20_trune_gumbel_things/VGG19truneTRY4/64921/start_for_trune800.h5')  # FULL it2

ds = datasets.cifar10()
ds['train'] = ds['train'].map(
    lambda x, y: (tfa.image.random_cutout(x, mask_size=6, constant_values=0), y))

optimizer = tf.optimizers.SGD(learning_rate=10, momentum=0.999, nesterov=True)
loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

kernels = [layer.kernel for layer in model.layers if hasattr(layer, "kernel")]
kernel_masks = [w for w in model.weights if "kernel_mask" in w.name]
bernoulli_distribs = [tf.Variable(mask * 25 - 15) for mask in kernel_masks]


# bernoulli_distribs = [tf.Variable(tf.ones_like(mask) * 2) for mask in kernel_masks]


def compare_mask_sets(masks1, masks2):
    M1 = set(np.where(np.concatenate([m1.reshape(-1) for m1 in masks1]))[0])
    M2 = set(np.where(np.concatenate([m1.reshape(-1) for m1 in masks2]))[0])
    print("rec", round(rec := len(M1 & M2) / len(M1), 3), end=" | ")
    print("prc", round(prc := len(M1 & M2) / len(M2), 3), end=" | ")
    print("F1", round(2 * prc * rec / (prc + rec), 3), end=" | ")
    print("acc", round(len(M1 & M2) / len(M1 | M2), 3))
    print()


def regularize_kernel_mask(layer):
    def _f():
        return tf.reduce_sum(layer.kernel_mask)

    return _f


regularization_losses = []
for layer in model.layers:
    if hasattr(layer, "kernel_mask"):
        regularization_losses.append(regularize_kernel_mask(layer))

acc_metric = tf.metrics.SparseCategoricalAccuracy()
loss_metric = tf.metrics.SparseCategoricalCrossentropy(True)


@tf.function
def train_step(x, y, decay):
    for kmask, distrib in zip(kernel_masks, bernoulli_distribs):
        clipped_mask = tf.sigmoid(distrib)
        # binary_mask = tf.random.uniform(shape=clipped_mask.shape)
        # binary_mask = tf.random.uniform(shape=clipped_mask.shape)
        # kmask.assign(tf.cast(binary_mask < clipped_mask, kmask.dtype))
        kmask.assign(clipped_mask)

    with tf.GradientTape() as tape:
        tape.watch(kernel_masks)
        outs = model(x)
        outs = tf.cast(outs, tf.float32)

        loss = loss_fn(y, outs)
        loss += tf.add_n([l() for l in regularization_losses]) * decay
        scaled_loss = loss * 256
    scaled_grads = tape.gradient(scaled_loss, kernel_masks)
    grads = [grad / 256 for grad in scaled_grads]
    loss_metric(y, outs)
    acc_metric(y, outs)

    updates = grads
    # for i, update in enumerate(updates):
    #     mask = tf.greater(update, 0)
    #     indices = tf.where(mask)
    #     tf.debugging.assert_integer(indices)
    #     updates[i] = tf.tensor_scatter_nd_update(
    #         update, indices, tf.ones([tf.shape(indices)[0]], dtype=update.dtype) * 0.1)
    #
    #     mask = tf.less(update, 0)
    #     indices = tf.where(mask)
    #     updates[i] = tf.tensor_scatter_nd_update(
    #         update, indices, tf.ones([tf.shape(indices)[0]], dtype=update.dtype) * -0.1)

    optimizer.apply_gradients(zip(updates, bernoulli_distribs))

    for mask in bernoulli_distribs:
        mask.assign(tf.clip_by_value(mask, -15, 15))


def set_expected_masks():
    for kmask, distrib in zip(kernel_masks, bernoulli_distribs):
        clipped_mask = tf.sigmoid(distrib)
        kmask.assign(tf.cast(0.5 < clipped_mask, kmask.dtype))


# @tf.function
def train_epoch(ds, decay, num_iter):
    progbar = tf.keras.utils.Progbar(num_iter)

    for x, y in ds.take(num_iter):
        train_step(x, y, decay)
        progbar.add(1)


@tf.function
def valid_epoch(ds=ds['valid']):
    for x, y in ds:
        # for kmask, distrib in zip(kernel_masks, bernoulli_distribs):
        #     clipped_mask = tf.sigmoid(distrib)
        #     binary_mask = tf.random.uniform(shape=clipped_mask.shape)
        #     kmask.assign(tf.cast(binary_mask < clipped_mask, kmask.dtype))

        outs = model(x, training=False)
        acc_metric(y, outs)
        loss_metric(y, outs)


def reset_metrics():
    acc = acc_metric.result()
    loss = loss_metric.result()
    acc_metric.reset_states()
    loss_metric.reset_states()
    return acc, loss


decay = tf.Variable(1e-7, trainable=False)

valid_epoch()
vacc, vloss = reset_metrics()
density = report_density(model)
apply_pruning_for_model(model)

print(f"EP {0}", f"DENSITY {density:6.4f}", f"VACC {vacc:6.4f}")
plt.hist(bernoulli_distribs[2].numpy().flatten(), bins=40)
plt.show()

# %%

num_iter = 400
for ep in range(32):
    t0 = time.time()
    train_epoch(ds['train'], decay, num_iter)
    tacc, tloss = reset_metrics()

    # set_expected_masks()

    valid_epoch()
    vacc, vloss = reset_metrics()
    density = report_density(model, detailed=True)

    print(
        f"EP {ep + 1}",
        f"DENSITY {density:7.5f}",
        f"VACC {vacc:7.5f}",
        f"TACC {tacc:7.5f}",
        f"TIME {time.time() - t0:7.2f}",
        sep=" | ",
    )

    model.save_weights("temp/classical_truning_on_train.h5")
    plt.hist(bernoulli_distribs[2].numpy().flatten(), bins=40)
    plt.show()

    # decay.assign_add((1e-6 - decay) / 2)

# %%

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tools import models, toolkit, utils, datasets, layers, pruning

ds = datasets.cifar10(repeat_train=False, shuffle_train=False)
model = models.VGG((32, 32, 3), n_classes=10, version=19)
# model.load_weights('data/VGG19_IMP03_ticket/130735/0.h5')
model.load_weights('temp/new_trune_workspace.h5')
# model.load_weights('temp/new_trune_workspace_destillation.h5')
# model.load_weights('data/VGG13_full/701455/0.h5')
# model.load_weights('data/VGG19_IMP03_ticket/770423/0.h5')
# model.load_weights('data/VGG19_IMP03_ticket/770423/1.h5')
# model.load_weights('data/VGG19_IMP03_ticket/770423/5.h5')
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# %%

method = 'truning'

sp2acc = []
sp = [
    *np.arange(0.0, 0.6, 0.2),
    *np.arange(0.6, 0.9, 0.02),
    *np.arange(0.9, 0.99, 0.01),
    *np.arange(0.99, 0.999, 0.001)
]

if method == 'truning':
    saliences = {}
    for layer in model.layers:
        if hasattr(layer, 'kernel_mask'):
            v = tf.sigmoid(layer.kernel_mask)
            saliences[layer.kernel.name] = v.numpy()
    toolkit.set_kernel_masks_values_on_model(model, 1)

for sparsity in sp:
    if method == 'magnitude':
        model = pruning.prune_l1(model,
                                 config=utils.ddict({'sparsity': sparsity}),
                                 silent=True)
    if method == 'truning':
        model = pruning.prune_by_saliences(model,
                                           saliences=saliences,
                                           config=utils.ddict({'sparsity': sparsity}),
                                           silent=False)
    pruning.apply_pruning_for_model(model)

    Acc = tf.keras.metrics.Mean()
    for x, y in ds.valid:
        loss, acc = toolkit.valid_step(x, y, model, loss_fn, training=True)
        Acc(tf.reduce_mean(acc))
    acc = Acc.result()

    print(f"Sparsity: {sparsity:6.3f}; Acc: {acc:6.3f}")
    sp2acc.append((sparsity, acc))
    if acc < 0.5:
        break

# %%

fig, ax = plt.subplots()
ax.grid()


# %%


def forward(x):
    return np.exp(x * 4)


def inverse(x):
    return np.log(x) / 4


ax.set_yscale('function', functions=(forward, inverse))
fig.show()

ax.set_ylim(0.5, 1.01)
ax.set_xlabel('1-shot pruning sparsity')
ax.set_title("Accuracy without retraining the network")

ax.plot(*zip(*sp2acc), linewidth=2, label='Magnitude pruning (VGG19)')
ax.legend()

ax.set_xlim(0.5, 0.995)
ax.set_xscale('linear')
fig.show()

# %%

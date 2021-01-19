import yaml
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import tools.toolkit as toolkit
import tools.models as models
import tools.datasets as datasets

# %%

logger = toolkit.Logger(column_width=10)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
ds = datasets.cifar10()

model = models.VGG((32, 32, 3), n_classes=10, version=19, l2_reg=1e-4)

# data = defaultdict(list)
names = []


def valid_step(x, y):
    outs = model(x, training=False)
    outs = tf.cast(outs, tf.float32)
    loss = loss_fn(y, outs)
    logger['valid_loss'](loss)
    logger['valid_acc'](tf.keras.metrics.sparse_categorical_accuracy(y, outs))


for exp in yaml.load_all(open('temp_logs.yaml', 'r')):
    if 'trune' in exp['name'].lower() and exp['num_iterations'] == 80000:
        acc = exp.get('acc') or exp.get('ACC')

        path = exp.get('full_path')
        if path[:4] == 'logs':
            path = 'temp' + path[4:]
        path = path + '.h5'

        if path in names:
            print(f'skipping repeat {path}')
            continue
        else:
            names.append(path)

        try:
            model.load_weights(path)
        except OSError:
            print(f'skipping {path}')
            continue

        km = toolkit.get_kernel_masks(model)
        km = np.concatenate([m.numpy().flatten() for m in km])
        sparsity = 1 - np.mean(km)
        if sparsity == 0:
            print(f'skipping not pruned {path}')
            continue

        # for x, y in ds['valid']:
        #     valid_step(x, y)
        acc = exp.get('acc') or exp.get('ACC')

        data['trune'].append([acc, sparsity, path])
        print(path, acc, sparsity)

for exp in yaml.load_all(open('temp_logs.yaml', 'r')):
    if 'snip' in exp['name'].lower() and exp['num_iterations'] == 80000:
        acc = exp.get('acc') or exp.get('ACC')

        path = exp.get('full_path')
        if path[:4] == 'logs':
            path = 'temp' + path[4:]
        path = path + '.h5'

        if path in names:
            print(f'skipping repeat {path}')
            continue
        else:
            names.append(path)

        try:
            model.load_weights(path)
        except OSError:
            print(f'skipping {path}')
            continue

        km = toolkit.get_kernel_masks(model)
        km = np.concatenate([m.numpy().flatten() for m in km])
        sparsity = 1 - np.mean(km)
        if sparsity == 0:
            print(f'skipping not pruned {path}')
            continue

        acc = exp.get('acc') or exp.get('ACC')

        data['snip'].append([acc, sparsity])
        print(path, acc, sparsity)

# %%

for exp in yaml.load_all(open('temp_logs.yaml', 'r')):
    if 'magnitude' in exp['name'].lower() and exp['num_iterations'] == 80000:
        acc = exp.get('acc') or exp.get('ACC')

        path = exp.get('full_path')
        if path[:4] == 'logs':
            path = 'temp' + path[4:]
        path = path + '.h5'

        if path in names:
            print(f'skipping repeat {path}')
            continue
        else:
            names.append(path)

        try:
            model.load_weights(path)
        except OSError:
            print(f'skipping {path}')
            continue

        km = toolkit.get_kernel_masks(model)
        km = np.concatenate([m.numpy().flatten() for m in km])
        sparsity = 1 - np.mean(km)
        if sparsity == 0:
            print(f'skipping not pruned {path}')
            continue

        acc = exp.get('acc') or exp.get('ACC')

        data['magnitude'].append([acc, sparsity])
        print(path, acc, sparsity)

# %%

for exp in yaml.load_all(open('imp_logs.yaml', 'r')):
    print(exp)
    if 'DENSITY' in exp:
        if exp['DENSITY'] < 0.2 and exp["ACC"] > 0.9:
            data['imp'].append([exp['ACC'], 1 - exp['DENSITY']])

# %%
plt.figure(figsize=(8, 5), dpi=200)

name2marker = {'trune': 'x',
               'snip': 'o',
               'imp': 'P',
               'magnitude': 'p'}

for name, d in data.items():
    npdata = np.array(d)[:, :2].astype(np.float)
    plt.scatter(npdata[:, 1], npdata[:, 0], marker=name2marker[name], label=name)
plt.legend()

plt.xscale('logit', one_half='')
xticks = [0.3, 0.75, 0.88, 0.938, 0.969, 0.984, 0.992, 0.996, 0.998, 0.999]
xnames = [round(x * 100, 1) for x in xticks]
plt.xticks(xticks, xnames)
plt.show()

# %%

data

# %%

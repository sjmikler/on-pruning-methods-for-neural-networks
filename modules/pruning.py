import os

import numpy as np
import tensorflow as tf

from tools import datasets, layers, models
from tools.utils import (
    contains_any,
    cprint,
    get_optimizer,
    logging_from_history,
    reset_weights_to_checkpoint,
)


def main(exp):
    """
    PROCEDURES IN ORDER:
    1. Creating model
    2. Loading checkpoint Before Pruning
    3. Applying pruning
    4. Loading checkpoint After Pruning
    5. Pruning related procedures After Pruning

    EXPERIMENT KEYS:
    * checkpointBP: not required
    * checkpointAP: not required
    """
    cprint("RUNNING PRUNING MODULE")
    globally_enable_pruning()

    ds = datasets.get_dataset(exp.dataset,
                              precision=exp.precision,
                              **exp.dataset_config)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model_func = models.get_model(exp.model)
    model_config = exp.model_config
    model = model_func(**model_config)

    optimizer = get_optimizer(exp.optimizer, exp.optimizer_config)
    model.compile(optimizer, loss_fn, metrics=["accuracy"])

    # load checkpointed weights before the pruning
    if hasattr(exp, 'checkpointBP'):
        model.load_weights(exp.checkpointBP)
        cprint(f"LOADED BEFORE PRUNING {exp.checkpointBP}")

    model = set_pruning_masks(model=model,
                              pruning_method=exp.pruning,
                              pruning_config=exp.pruning_config,
                              dataset=ds, )
    assert isinstance(model, tf.keras.Model)

    # load or reset weights after the pruning, do not change masks
    if hasattr(exp, 'checkpointAP'):
        if exp.checkpointAP == 'random':
            ckp = None
        else:
            ckp = exp.checkpointAP
        num_masks = reset_weights_to_checkpoint(model,
                                                ckp=ckp,
                                                skip_keyword='kernel_mask')
        cprint(f"LOADED AFTER PRUNING {exp.checkpointAP}, but keeping {num_masks} "
               f"masks!")

    # apply pruning from previously calculated masks
    apply_pruning_masks(model, pruning_method=exp.pruning)
    steps_per_epoch = min(exp.steps, exp.steps_per_epoch)

    if unused := exp.get_unused_parameters():
        cprint("WARNING! Listing unused parameters:", color='red')
        cprint(unused, color='red')

    if exp.steps != 0:
        history = model.fit(x=ds.train,
                            validation_data=ds.valid,
                            steps_per_epoch=steps_per_epoch,
                            epochs=int(exp.steps / steps_per_epoch), )
        info = exp.copy()
        info["FINAL_DENSITY"] = report_density(model)
        cprint("FINAL DENSITY:", info["FINAL_DENSITY"])
        logging_from_history(history.history, info=info)

    os.makedirs(os.path.dirname(exp.checkpoint), exist_ok=True)
    model.save_weights(exp.checkpoint, save_format="h5")


# %%

# %%

def globally_enable_pruning():
    tf.keras.layers.Dense = layers.MaskedDense
    tf.keras.layers.Conv2D = layers.MaskedConv
    cprint("PRUNING IS ENABLED GLOBALLY! LAYERS HAVE BEEN REPLACED...")


# %%


def structurize_matrix(matrix, n_clusters):
    shape = matrix.shape
    if len(shape) == 2:
        return structurize_dense(matrix, n_clusters)
    elif len(shape) == 4:
        return structurize_conv(matrix, n_clusters)
    else:
        raise Exception


def structurize_dense(matrix, n_clusters):
    """
    Finds similar structurized matirx with `n_clusters` unique rows
    :param matrix: numpy 2-dimensional array
    :param n_clusters: int, number of clusters to find
    :return: structured array, the same shape and dtype and matrix
    """
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters)

    clusters = kmeans.fit_predict(matrix)
    centers = kmeans.cluster_centers_

    new_matrix = np.take(centers, clusters, axis=0)
    return new_matrix


def structurize_conv(matrix, n_clusters):
    if matrix.shape[2] < n_clusters:
        return matrix

    matrix = np.moveaxis(matrix, 2, 0)
    org_shape = matrix.shape
    matrix = matrix.reshape(matrix.shape[0], -1)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters)

    clusters = kmeans.fit_predict(matrix)
    centers = kmeans.cluster_centers_

    new_matrix = np.take(centers, clusters, axis=0)
    new_matrix = new_matrix.reshape(org_shape)
    new_matrix = np.moveaxis(new_matrix, 0, 2)
    return new_matrix


def structurize_anything(structure, saliences):
    if structure is True:
        cprint(f"NEURON-WISE STRUCTURING!")
        saliences = l1_saliences_over_channel(saliences)
    elif isinstance(structure, int):
        cprint(f"ENFORCING {structure} GROUPS!")
        saliences = {key: structurize_matrix(value, structure) for key, value in
                     saliences.items()}
    return saliences


# %%

def snip_saliences(model, loader, batches=1):
    """
    :param model: callable model with trainable_weights
    :param loader: `tf.data.Dataset` with `.take` method
    :param batches: int, number of batches to take with `.take` method
    :return: dict, keys are Variable name, values are saliences from SNIP
    """
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    cumulative_grads = [tf.zeros_like(w) for w in model.trainable_weights]

    for x, y in loader.take(batches):
        with tf.GradientTape() as tape:
            outs = model(x)
            outs = tf.cast(outs, tf.float32)
            loss = loss_fn(y, outs)
        grads = tape.gradient(loss, model.trainable_weights)
        cumulative_grads = [c + g for c, g in zip(cumulative_grads, grads)]
    saliences = {w.name: tf.abs(w * g).numpy() for w, g in
                 zip(model.trainable_weights, cumulative_grads)}
    return saliences


def psuedo_snip_saliences(model, *args, **kwargs):
    """
    :param model: callable model with trainable_weights
    :return: dict, keys are Variable name, values are saliences from SNIP
    """
    cumulative_grads = [tf.random.uniform(shape=w.shape, minval=-1, maxval=1) for w in
                        model.trainable_weights]
    saliences = {w.name: tf.abs(w * g).numpy() for w, g in
                 zip(model.trainable_weights, cumulative_grads)}
    return saliences


def grasp_saliences(model, loader, batches=1):
    """
    :param model: callable model with trainable_weights
    :param loader: `tf.data.Dataset` with `.take` method
    :param batches: int, number of batches to take with `.take` method
    :return: dict, keys are Variable name, values are saliences from GraSP
    """
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    cumulative_grads = [tf.zeros_like(w) for w in model.trainable_weights]

    for x, y in loader.take(batches):
        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape2:
                outs = model(x)
                outs = tf.cast(outs, tf.float32)
                loss = loss_fn(y, outs)
            g1 = tape2.gradient(loss, model.trainable_weights)
            g1 = tf.concat([tf.reshape(g, -1) for g in g1], 0)
            g1 = tf.reduce_sum(g1 * tf.stop_gradient(g1))
        g2 = tape.gradient(g1, model.trainable_weights)
        cumulative_grads = [c + g for c, g in zip(cumulative_grads, g2)]

    saliences = {w.name: -(w * g).numpy() for w, g in
                 zip(model.trainable_weights, cumulative_grads)}
    return saliences


def minus_grasp_saliences(model, loader, batches=1):
    """
    :param model: callable model with trainable_weights
    :param loader: `tf.data.Dataset` with `.take` method
    :param batches: int, number of batches to take with `.take` method
    :return: dict, keys are Variable name, values are saliences from GraSP
    """
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    cumulative_grads = [tf.zeros_like(w) for w in model.trainable_weights]

    for x, y in loader.take(batches):
        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape2:
                outs = model(x)
                outs = tf.cast(outs, tf.float32)
                loss = loss_fn(y, outs)
            g1 = tape2.gradient(loss, model.trainable_weights)
            g1 = tf.concat([tf.reshape(g, -1) for g in g1], 0)
            g1 = tf.reduce_sum(g1 * tf.stop_gradient(g1))
        g2 = tape.gradient(g1, model.trainable_weights)
        cumulative_grads = [c + g for c, g in zip(cumulative_grads, g2)]

    saliences = {w.name: (w * g).numpy() for w, g in
                 zip(model.trainable_weights, cumulative_grads)}
    return saliences


def get_pruning_mask(saliences, percentage):
    """
    :param saliences: list of saliences arrays
    :param percentage: at most this many percentage of weights will be zeroed
    :return: list of masks of 1's and 0's with corresponding sizes
    """
    sizes = [w.size for w in saliences]
    shapes = [w.shape for w in saliences]

    flatten = np.concatenate([w.reshape(-1) for w in saliences])
    flat_mask = np.ones_like(flatten)

    threshold = np.percentile(flatten, percentage * 100)
    cprint(f'pruning threshold: {threshold:8.5f}')
    flat_mask[flatten < threshold] = 0

    cumsizes = np.cumsum(sizes)[:-1]
    flat_masks = np.split(flat_mask, cumsizes)
    return [w.reshape(shape) for w, shape in zip(flat_masks, shapes)]


def saliences2masks(saliences_dict, percentage):
    """
    :param saliences_dict: keys are variable names, values are saliences
    :param percentage: float from 0 to 1
    :return: dict, keys are variable names, values are masks
    """
    saliences = list(saliences_dict.values())
    masks = get_pruning_mask(saliences, percentage)
    return {key: mask for key, mask in zip(saliences_dict, masks)}


def extract_kernels(dictionary):
    return {key: value for key, value in dictionary.items() if "kernel" in key}


# %%

def set_kernel_masks_for_model(model, masks_dict, silent=False):
    for mask in masks_dict:
        for layer in model.layers:
            for weight in layer.weights:
                if mask == weight.name:
                    layer.set_pruning_mask(masks_dict[mask])
                    if not silent:
                        cprint(f"pruning {weight.name} to {layer.sparsity * 100:.2f}%")
                        cprint(f"left in {weight.name} : {layer.left_unpruned}")


def l1_saliences_over_channel(saliences):
    for key, value in saliences.items():
        org_shape = value.shape
        if len(org_shape) == 2:
            value = np.abs(value)
            value = np.mean(value, 1)
            value = np.repeat(np.expand_dims(value, 1), org_shape[1], axis=1)
            saliences[key] = value

        elif len(org_shape) == 4:
            value = np.abs(value)
            value = np.mean(value, 3)
            value = np.repeat(np.expand_dims(value, 3), org_shape[3], axis=3)
            saliences[key] = value

        else:
            raise TypeError('Weight matrix should have 2 or 4 dimensions!')
    return saliences


# %%

def prune_by_kernel_masks(model, config, silent=False):
    sparsity = config.sparsity
    structure = config.get('structure')
    saliences = {}
    for layer in model.layers:
        if hasattr(layer, 'kernel_mask'):
            kernel = layer.kernel
            saliences[kernel.name] = layer.kernel_mask.numpy()
    saliences = extract_kernels(saliences)
    if structure:
        saliences = structurize_anything(structure, saliences)
    masks = saliences2masks(saliences, percentage=sparsity)
    set_kernel_masks_for_model(model, masks, silent)
    return model


def prune_GraSP(model, dataset, config, silent=False):
    """Hessian related pruning."""

    raise NotImplementedError("Check if the implementation is correct!")

    sparsity = config.sparsity
    n_batches = config.n_batches or 1
    structure = config.get('structure')

    saliences = grasp_saliences(model, dataset, batches=n_batches)
    saliences = extract_kernels(saliences)
    if structure:
        saliences = structurize_anything(structure, saliences)
    masks = saliences2masks(saliences, percentage=sparsity)
    set_kernel_masks_for_model(model, masks, silent)
    return model


def prune_SNIP(model, dataset, config, silent=False):
    """Prune by saliences `|W*G|` for W being weights an G being gradients."""

    sparsity = config.sparsity
    n_batches = config.get('n_batches') or 1
    structure = config.get('structure')

    saliences = snip_saliences(model, dataset, batches=n_batches)
    saliences = extract_kernels(saliences)
    if structure:
        saliences = structurize_anything(structure, saliences)
    masks = saliences2masks(saliences, percentage=sparsity)
    set_kernel_masks_for_model(model, masks, silent)
    return model


def prune_pseudo_SNIP(model, dataset, config, silent=False):
    """In SNIP's `W*G` we replace gradients G with a random from [-1, 1]."""

    sparsity = config.sparsity
    structure = config.get('structure')

    saliences = psuedo_snip_saliences(model)
    saliences = extract_kernels(saliences)
    if structure:
        saliences = structurize_anything(structure, saliences)
    masks = saliences2masks(saliences, percentage=sparsity)
    set_kernel_masks_for_model(model, masks, silent)
    return model


def prune_random(model, config, silent=False):
    """Random, non-uniform pruning."""

    sparsity = config.sparsity
    structure = config.get('structure')
    saliences = {w.name: np.random.rand(*w.shape) for w in model.trainable_weights}
    saliences = extract_kernels(saliences)
    if structure:
        saliences = structurize_anything(structure, saliences)
    masks = saliences2masks(saliences, percentage=sparsity)
    set_kernel_masks_for_model(model, masks, silent)
    return model


def prune_l1(model, config, silent=False):
    """Prune smallest magnitudes."""

    sparsity = config.sparsity
    structure = config.get('structure')
    saliences = {w.name: np.abs(w.numpy()) for w in model.trainable_weights}
    saliences = extract_kernels(saliences)
    if structure:
        saliences = structurize_anything(structure, saliences)
    masks = saliences2masks(saliences, percentage=sparsity)
    set_kernel_masks_for_model(model, masks, silent)
    return model


# %%

def shuffle_masks(model):
    """Keep weights intact, shuffle masks inside layers."""

    for layer in model.layers:
        if hasattr(layer, "kernel_mask"):
            mask = layer.kernel_mask.numpy()
            mask_shape = mask.shape
            mask = mask.reshape(-1)
            np.random.shuffle(mask)
            mask = mask.reshape(mask_shape)
            layer.kernel_mask.assign(mask)
    return model


def shuffle_weights(model):
    """Keep masks intact, shuffle nonzero weights inside layers."""

    for layer in model.layers:
        if hasattr(layer, "kernel_mask"):
            kernel = layer.kernel.numpy()
            mask = layer.kernel_mask.numpy().astype(np.bool)
            kernel_nonzero = kernel[mask]
            np.random.shuffle(kernel_nonzero)
            kernel[mask] = kernel_nonzero
            layer.kernel.assign(kernel)
    return model


def shuffle_layers(model):
    """Shuffle both weights and masks (glued) inside layers."""

    for layer in model.layers:
        if hasattr(layer, "kernel_mask"):
            mask = layer.kernel_mask.numpy()
            kern = layer.kernel.numpy()
            shape = mask.shape

            mask = mask.reshape(-1)
            kern = kern.reshape(-1)
            perm = np.random.permutation(mask.size)
            mask = mask[perm]
            kern = kern[perm]

            mask = mask.reshape(shape)
            kern = kern.reshape(shape)
            layer.kernel_mask.assign(mask)
            layer.kernel.assign(kern)
    return model


# %%

def report_density(model, detailed=False):
    nonzero = 0
    kernels = 0
    biases = 0
    for w in model.weights:
        if 'kernel_mask' in w.name:
            km = w.numpy()
            kernels += km.size
            nonzero_here = (km != 0).sum()
            nonzero += nonzero_here
            if detailed:
                cprint(f"density of {w.name:>16}: {nonzero_here / km.size:6.4f}")
        if 'bias' in w.name or 'beta' in w.name:
            biases += w.shape.num_elements()
    if detailed:
        cprint(f"Biases make {biases / (kernels + biases) * 100:6.3f}% of weights!")
    if kernels == 0:
        return 1.0
    return nonzero / kernels


# %%

def initialize_kernel_masks(model):
    nmasks = 0
    for layer in model.layers:
        if type(layer) in [tf.keras.layers.Conv2D, tf.keras.layers.Dense]:
            layer.kernel_mask = layer.add_weight(name=layer.kernel.name.replace(
                'kernel',
                'kernel_mask'),
                shape=layer.kernel.shape,
                dtype=layer.kernel.dtype,
                initializer="ones",
                trainable=False, )
            nmasks += layer.kernel.shape.num_elements()
    cprint(f"CREATED {nmasks} KERNEL MASKS!")


def set_pruning_masks(model, pruning_method, pruning_config, dataset):
    if contains_any(pruning_method.lower(), 'none', 'nothing'):
        cprint('NO PRUNING')
        return model
    elif contains_any(pruning_method.lower(), 'random'):
        cprint('RANDOM PRUNING')
        model = prune_random(model=model, config=pruning_config)
    elif contains_any(pruning_method.lower(), 'snip'):
        if contains_any(pruning_method.lower(), 'pseudo'):
            cprint('PSUEDO SNIP PRUNING')
            model = prune_pseudo_SNIP(model=model,
                                      config=pruning_config,
                                      dataset=dataset.train)
        else:
            cprint('SNIP PRUNING')
            model = prune_SNIP(model=model,
                               config=pruning_config,
                               dataset=dataset.train)
    elif contains_any(pruning_method.lower(), 'grasp'):
        cprint('GRASP PRUNING')
        model = prune_GraSP(model=model, config=pruning_config, dataset=dataset.train)
    elif contains_any(pruning_method.lower(), 'l1', 'magnitude'):
        cprint('WEIGHT MAGNITUDE PRUNING')
        model = prune_l1(model=model, config=pruning_config)
    elif contains_any(pruning_method.lower(), 'imp_complete'):
        cprint('IMP COMPLETE PRUNING')
        model = prune_IMP_complete(model=model, config=pruning_config)
    elif contains_any(pruning_method.lower(), 'kernel mask'):
        cprint("PRUNING BY KERNEL MASK VALUES")
        model = prune_by_kernel_masks(model=model, config=pruning_config)
    else:
        raise KeyError(f"PRUNING {pruning_method} is unknown!")
    return model


def apply_pruning_for_model(model):
    """Set masked weights to 0."""

    for layer in model.layers:
        if hasattr(layer, "apply_pruning_mask"):
            layer.apply_pruning_mask()


def apply_pruning_masks(model, pruning_method):
    """Wrapper for `apply_pruning_for_model`"""

    if contains_any(pruning_method.lower(), 'shuffle weight'):
        cprint("SHUFFLING WEIGHTS IN LAYERS!")
        model = shuffle_weights(model=model)

    if contains_any(pruning_method.lower(), 'shuffle layer'):
        cprint("SHUFFLING WEIGHTS WITH MASKS IN LAYERS!")
        model = shuffle_layers(model=model)

    if contains_any(pruning_method.lower(), 'shuffle mask'):
        cprint("SHUFFLING MASKS IN LAYERS!")
        model = shuffle_masks(model=model)

    apply_pruning_for_model(model)
    density = report_density(model, detailed=True)
    cprint(f"REPORTING DENSITY: {density:7.5f}")
    return model


# %%

def get_kernel_masks(model):
    return [l.kernel_mask for l in model.layers if hasattr(l, 'kernel_mask')]


def set_kernel_masks_values_on_model(model, values):
    for i, kernel in enumerate(get_kernel_masks(model)):
        if isinstance(values, int) or isinstance(values, float):
            mask = np.ones_like(kernel.numpy()) * values
        else:
            mask = values[i]
        kernel.assign(mask)


def set_kernel_masks_values(masks, values):
    if isinstance(values, int) or isinstance(values, float):
        for mask in masks:
            mask.assign(np.ones_like(mask.numpy()) * values)
    else:
        for mask, value in zip(masks, values):
            mask.assign(value)


def set_kernel_masks_object(model, masks):
    layers = (l for l in model.layers if hasattr(l, 'kernel_mask'))
    for l, km in zip(layers, masks):
        l.kernel_mask = km

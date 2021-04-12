import numpy as np
import tensorflow as tf

from tools.utils import contains_any


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


def reset_weights_to_checkpoint(model, ckp=None, skip_keyword=None):
    """Reset network in place, ability to skip keybword."""

    temp = tf.keras.models.clone_model(model)
    if ckp:
        temp.load_weights(ckp)
    skipped = 0
    for w1, w2 in zip(model.weights, temp.weights):
        if skip_keyword in w1.name:
            skipped += 1
            continue
        w1.assign(w2)
    print(
        f"INFO RESET: Skipped {skipped} layers with keyword {skip_keyword}!")
    return skipped


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
    saliences = {
        w.name: tf.abs(w * g).numpy()
        for w, g in zip(model.trainable_weights, cumulative_grads)
    }
    return saliences


def psuedo_snip_saliences(model, *args, **kwargs):
    """
    :param model: callable model with trainable_weights
    :return: dict, keys are Variable name, values are saliences from SNIP
    """
    cumulative_grads = [tf.random.uniform(shape=w.shape, minval=-1, maxval=1)
                        for w in model.trainable_weights]
    saliences = {
        w.name: tf.abs(w * g).numpy()
        for w, g in zip(model.trainable_weights, cumulative_grads)
    }
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

    saliences = {w.name: -(w * g).numpy()
                 for w, g in zip(model.trainable_weights, cumulative_grads)}
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

    saliences = {w.name: (w * g).numpy()
                 for w, g in zip(model.trainable_weights, cumulative_grads)}
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
    print(f'pruning threshold: {threshold:8.5f}')
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
                        print(f"pruning {weight.name} to {layer.sparsity * 100:.2f}%")
                        print(f"left in {weight.name} : {layer.left_unpruned}")


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

def structurize_anything(structure, saliences):
    if structure is True:
        print(f"NEURON-WISE STRUCTURING!")
        saliences = l1_saliences_over_channel(saliences)
    elif isinstance(structure, int):
        print(f"ENFORCING {structure} GROUPS!")
        saliences = {
            key: structurize_matrix(value, structure) for key, value in
            saliences.items()
        }
    return saliences


def prune_by_kernel_masks(model, config, silent=False):
    sparsity = config.sparsity
    structure = config.structure
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

    sparsity = config.sparsity
    n_batches = config.n_batches or 1
    structure = config.structure

    saliences = grasp_saliences(model, dataset, batches=n_batches)
    saliences = extract_kernels(saliences)
    if structure:
        saliences = structurize_anything(structure, saliences)
    masks = saliences2masks(saliences, percentage=sparsity)
    set_kernel_masks_for_model(model, masks, silent)
    return model


def prune_TRUNE(model, dataset, config, silent=False):
    """Experimental, learnable pruning. Very early version."""

    print(config)
    learning_rate = config.learning_rate
    momentum = config.momentum
    weight_decay = float(config.weight_decay)
    num_iterations = config.num_iterations
    steps_per_epoch = config.steps_per_epoch

    from experimental.tools.trune import truning
    model = truning(model, learning_rate, momentum, weight_decay, num_iterations,
                    steps_per_epoch,
                    dataset=dataset)
    set_kernel_masks_for_model(model,
                               masks_dict={layer.kernel.name: layer.kernel_mask.numpy()
                                           for layer in model.layers if
                                           hasattr(layer, 'kernel_mask')},
                               silent=silent)
    return model


def prune_SNIP(model, dataset, config, silent=False):
    """Prune by saliences `|W*G|` for W being weights an G being gradients."""

    sparsity = config.sparsity
    n_batches = config.n_batches or 1
    structure = config.structure

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
    n_batches = config.n_batches or 1
    structure = config.structure

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
    structure = config.structure
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
    structure = config.structure
    saliences = {w.name: np.abs(w.numpy()) for w in model.trainable_weights}
    saliences = extract_kernels(saliences)
    if structure:
        saliences = structurize_anything(structure, saliences)
    masks = saliences2masks(saliences, percentage=sparsity)
    set_kernel_masks_for_model(model, masks, silent)
    return model


def prune_IMP_complete(model, config):
    """Like l1 pruning, but contains checkpointAP step with additional options.

    Deprecated
    """

    saliences = {w.name: np.abs(w.numpy()) for w in model.trainable_weights}
    saliences = extract_kernels(saliences)
    if config.structure:
        saliences = structurize_anything(config.structure, saliences)
    masks = saliences2masks(saliences, percentage=config.sparsity)
    set_kernel_masks_for_model(model, masks)

    if config.get('checkpointAP'):
        signs = []
        for w in model.weights:
            sign = w.numpy()
            sign[sign >= 0] = 1
            sign[sign < 0] = -1
            signs.append(sign)

        reset_weights_to_checkpoint(model, config.checkpointBP,
                                    skip_keyword="kernel_mask")
        if config.get('preserve_BP_sign'):
            for w1, sign in zip(model.weights, signs):
                w1.assign(np.abs(w1.numpy()) * sign)
    return model


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


def apply_pruning_for_model(model):
    """Set masked weights to 0."""

    for layer in model.layers:
        if hasattr(layer, "apply_pruning_mask"):
            layer.apply_pruning_mask()


def report_density(model, detailed=False):
    nonzero = 0
    max_nonzero = 0
    for w in model.weights:
        if 'kernel_mask' in w.name:
            km = w.numpy()
            max_nonzero += km.size
            nonzero += (km != 0).sum()
            if detailed:
                print(f"density of {w.name:>16}: {km.sum() / km.size:6.4f}")

    if max_nonzero == 0:
        return 1.0
    return nonzero / max_nonzero


# %%

def set_pruning_masks(model,
                      pruning_method,
                      pruning_config,
                      dataset):
    if contains_any(pruning_method.lower(), 'none', 'nothing'):
        print('NO PRUNING')
        return model
    elif contains_any(pruning_method.lower(), 'random'):
        print('RANDOM PRUNING')
        model = prune_random(model=model,
                             config=pruning_config)
    elif contains_any(pruning_method.lower(), 'snip'):
        if contains_any(pruning_method.lower(), 'pseudo_snip'):
            print('PSUEDO SNIP PRUNING')
            model = prune_pseudo_SNIP(model=model,
                                      config=pruning_config,
                                      dataset=dataset.train)
        else:
            print('SNIP PRUNING')
            model = prune_SNIP(model=model,
                               config=pruning_config,
                               dataset=dataset.train)
    elif contains_any(pruning_method.lower(), 'grasp'):
        print('GRASP PRUNING')
        model = prune_GraSP(model=model,
                            config=pruning_config,
                            dataset=dataset.train)
    elif contains_any(pruning_method.lower(), 'l1', 'magnitude'):
        print('WEIGHT MAGNITUDE PRUNING')
        model = prune_l1(model=model,
                         config=pruning_config)
    elif contains_any(pruning_method.lower(), 'imp_complete'):
        print('IMP COMPLETE PRUNING')
        model = prune_IMP_complete(model=model,
                                   config=pruning_config)
    elif contains_any(pruning_method.lower(), 'trune'):
        print('TRUNE')
        model = prune_TRUNE(model=model,
                            config=pruning_config,
                            dataset=dataset)
    elif contains_any(pruning_method.lower(), 'kernel mask'):
        print("PRUNING BY KERNEL MASK VALUES")
        model = prune_by_kernel_masks(model=model,
                                      config=pruning_config)
    else:
        raise KeyError(f"PRUNING {pruning_method} is unknown!")
    return model


def apply_pruning_masks(model,
                        pruning_method):
    if contains_any(pruning_method.lower(), 'shuffle weight'):
        print("SHUFFLING WEIGHTS IN LAYERS!")
        model = shuffle_weights(model=model)

    if contains_any(pruning_method.lower(), 'shuffle layer'):
        print("SHUFFLING WEIGHTS WITH MASKS IN LAYERS!")
        model = shuffle_layers(model=model)

    if contains_any(pruning_method.lower(), 'shuffle mask'):
        print("SHUFFLING MASKS IN LAYERS!")
        model = shuffle_masks(model=model)

    apply_pruning_for_model(model)
    density = report_density(model)
    print(f"REPORTING DENSITY: {density:7.5f}")
    return model

# %%

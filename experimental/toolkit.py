import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from copy import deepcopy


def get_kernel_masks(model):
    return [w for w in model.weights if 'kernel_mask' in w.name]


def get_kernels(model):
    return [l.kernel for l in model.layers if hasattr(l, 'kernel')]


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


def set_all_weights_from_model(model, source_model):
    for w1, w2 in zip(model.weights, source_model.weights):
        if w1.shape == w2.shape:
            w1.assign(w2)
        else:
            print(f"skipping {w1.name}: {w1.shape} != {w2.shape}")


def clone_model(model):
    new_model = tf.keras.models.clone_model(model)
    set_all_weights_from_model(new_model, model)
    return new_model


def reset_weights_to_checkpoint(model, ckp, skip_keyword=None):
    temp = tf.keras.models.clone_model(model)
    temp.load_weights(ckp)
    for w1, w2 in zip(model.weights, temp.weights):
        if skip_keyword in w1.name:
            continue
        w1.assign(w2)


def clip_many(values, clip_at, clip_from=None, inplace=False):
    if clip_from is None:
        clip_from = -clip_at

    if inplace:
        for v in values:
            v.assign(tf.clip_by_value(v, clip_from, clip_at))
    else:
        r = []
        for v in values:
            r.append(tf.clip_by_value(v, clip_from, clip_at))
        return r


def concatenate_flattened(arrays):
    return np.concatenate([x.flatten() for x in arrays], axis=0)


def visualize_masks(masks, mask_activation):
    fig, axes = plt.subplots(5, 1, figsize=(7, 20), constrained_layout=True)
    activated_masks = [mask_activation(mask).numpy() for mask in masks]
    masks = [mask.numpy() for mask in masks]

    axes[0].set_title('4 layers not activated')
    axes[0].hist(concatenate_flattened(masks[:4]), bins=30)
    axes[1].set_title('4 layers activated')
    axes[1].hist(concatenate_flattened(activated_masks[:4]), bins=30)
    axes[2].set_title('all layers not activated')
    axes[2].hist(concatenate_flattened(masks), bins=30)
    axes[3].set_title('all layers activated')
    axes[3].hist(concatenate_flattened(activated_masks), bins=30)

    means = [np.mean(np.abs(mask)) for mask in activated_masks]
    axes[4].set_title('density of layers')
    axes[4].bar(range(len(means)), means)
    fig.show()
    return fig


def update_mask_info(kernel_masks, mask_activation, logger=None):
    mask = np.concatenate([
        tf.abs(mask_activation(mask)).numpy().flatten()
        for mask in kernel_masks
    ])
    if logger:
        logger['avg_mask'] = np.mean(mask)
        # logger['mask_std'] = np.std(mask)
    return mask


def compare_masks(perf_m, m, mask_activation, force_sparsity=None):
    from sklearn.metrics import precision_recall_curve
    m = np.concatenate([mask_activation(x).numpy().flatten() for x in m])
    m = np.abs(m)

    perf_m = np.concatenate([x.numpy().flatten() for x in perf_m])

    prc, rec, thr = precision_recall_curve(perf_m, m)
    f1_scores = [2 * p * r / (p + r) if (p + r) else -1 for p, r in zip(prc, rec)]
    idx = np.argmax(f1_scores)

    if force_sparsity:  # modify `idx` so sparsity is as required
        threshold = np.sort(m)[int(len(m) * force_sparsity)]
        for idx, t in enumerate(thr):
            if t > threshold:
                break

    f1_density = np.mean(m >= thr[idx])
    return f1_scores[idx], prc[idx], rec[idx], thr[idx], f1_density


def create_layers(mask_activation):
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

        def call(self, x):
            multipler = mask_activation(self.kernel_mask)
            masked_w = tf.multiply(self.kernel, multipler)
            result = tf.matmul(x, masked_w)

            if self.use_bias:
                result = tf.add(result, self.bias)
            return self.activation(result)

        def set_pruning_mask(self, new_mask: np.ndarray):
            tf.assert_equal(new_mask.shape, self.kernel_mask.shape)
            self.kernel_mask.assign(new_mask)

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

        def call(self, x):
            multipler = mask_activation(self.kernel_mask)
            masked_w = tf.multiply(self.kernel, multipler)
            result = tf.nn.conv2d(x, masked_w, strides=self.strides, padding=self.padding.upper())

            if self.use_bias:
                result = tf.add(result, self.bias)
            return self.activation(result)

        def set_pruning_mask(self, new_mask: np.ndarray):
            tf.assert_equal(new_mask.shape, self.kernel_mask.shape)
            self.kernel_mask.assign(new_mask)

        def apply_pruning_mask(self):
            self.kernel.assign(tf.multiply(self.kernel, self.kernel_mask))

    return MaskedConv, MaskedDense


def create_layers_vsign(mask_activation):
    class MaskedDense(tf.keras.layers.Dense):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def build(self, input_shape):
            super().build(input_shape)
            self.kernel_mask = self.add_weight(
                name="kernel_mask",
                shape=[2, *self.kernel.shape],
                dtype=self.kernel.dtype,
                initializer="ones",
                trainable=False,
            )

        def call(self, x):
            mask = mask_activation(self.kernel_mask)
            # mask = multipler[0] * (multipler[1] * 2 - 1)
            # mask = multipler[0] - multipler[1]

            masked_w = tf.multiply(self.kernel, mask)
            result = tf.matmul(x, masked_w)

            if self.use_bias:
                result = tf.add(result, self.bias)
            return self.activation(result)

        def set_pruning_mask(self, new_mask: np.ndarray):
            tf.assert_equal(new_mask.shape, self.kernel_mask.shape)
            self.kernel_mask.assign(new_mask)

        def apply_pruning_mask(self):
            self.kernel.assign(tf.multiply(self.kernel, self.kernel_mask))

    class MaskedConv(tf.keras.layers.Conv2D):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def build(self, input_shape):
            super().build(input_shape)
            self.kernel_mask = self.add_weight(
                name="kernel_mask",
                shape=[2, *self.kernel.shape],
                dtype=self.kernel.dtype,
                initializer="ones",
                trainable=False,
            )

        def call(self, x):
            mask = mask_activation(self.kernel_mask)
            # mask = multipler[0] * (multipler[1] * 2 - 1)
            # mask = multipler[0] - multipler[1]

            masked_w = tf.multiply(self.kernel, mask)
            result = tf.nn.conv2d(x, masked_w, strides=self.strides, padding=self.padding.upper())

            if self.use_bias:
                result = tf.add(result, self.bias)
            return self.activation(result)

        def set_pruning_mask(self, new_mask: np.ndarray):
            tf.assert_equal(new_mask.shape, self.kernel_mask.shape)
            self.kernel_mask.assign(new_mask)

        def apply_pruning_mask(self):
            self.kernel.assign(tf.multiply(self.kernel, self.kernel_mask))

    return MaskedConv, MaskedDense


def prune_and_save_model(net, mask_activation, threshold, path):
    nonzero = 0
    num_weights = 0
    model = clone_model(net)

    for l in model.layers:
        if not hasattr(l, 'kernel_mask'):
            continue
        mask = tf.cast(mask_activation(l.kernel_mask) > threshold, tf.float32)
        nonzero += np.sum(np.abs(mask.numpy()) == 1)
        num_weights += mask.numpy().size
        l.kernel_mask.assign(mask)
    print(f"Saving model with density {nonzero / num_weights:6.4f} as {path}")
    model.save_weights(path)
    return model


# %%


class Logger:
    def __init__(self, column_width):
        self.last_header = None
        self.cw = column_width
        self.data = {}
        self.skip = []

    def __getitem__(self, item):
        if item not in self.data:
            self.data[item] = tf.keras.metrics.Mean()
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def get(self, key):
        v = self.data[key]
        if hasattr(v, 'result'):
            return v.result().numpy()
        else:
            return v

    def show_header(self, column_width=None):
        if column_width is not None:
            self.cw = column_width

        p = []
        for key, value in self.data.items():
            if key in self.skip:
                continue

            if len(key) > self.cw:
                p.append(f"{key[:((self.cw - 1) // 2 + (self.cw - 1) % 2)].upper()}-"
                         f"{key[-((self.cw - 1) // 2):].upper()}")
            else:
                p.append(f"{key.upper().ljust(self.cw)}")
        print(*p, sep=' | ')
        self.last_header = (self.cw, list(self.data))

    def show(self, reset=True):
        results = {}
        for key, value in self.data.items():
            if key in self.skip:
                continue

            if hasattr(value, 'result'):
                results[key] = value.result().numpy()
                if reset:
                    value.reset_states()
            else:
                results[key] = value
        p = []
        for key, value in results.items():
            p.append(f"{str(value)[:self.cw].ljust(self.cw)}")
        if self.last_header != (self.cw, list(self.data)):
            self.show_header()
        print(*p, sep=' | ')

    def omit_showing(self, *keys):
        if not keys:
            return

        for key in keys:
            self.skip.append(key)

    def reset_omitting(self):
        self.skip = []

    def reset(self, *keys):
        if not keys:
            keys = self.data.keys()

        for key in keys:
            value = self.data[key]
            if hasattr('reset_states'):
                value.reset_states()

    def peek(self, *keys):
        if not keys:
            keys = self.data.keys()

        results = {}
        for key in keys:
            value = self.data.get(key)
            if hasattr(value, 'result'):
                results[key] = value.result().numpy()
            else:
                results[key] = value
        return results

    def clear(self):
        self.data = {}

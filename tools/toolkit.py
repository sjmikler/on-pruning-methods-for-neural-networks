import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def get_kernel_masks(model):
    return [l.kernel_mask for l in model.layers if hasattr(l, 'kernel_mask')]


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
    """Warning if a pair doesn't match."""

    for w1, w2 in zip(model.weights, source_model.weights):
        if w1.shape == w2.shape:
            w1.assign(w2)
        else:
            print(f"WARNING: Skipping {w1.name}: {w1.shape} != {w2.shape}")


def clone_model(model):
    """tf.keras.models.clone_model + toolkit.set_all_weights_from_model"""

    new_model = tf.keras.models.clone_model(model)
    set_all_weights_from_model(new_model, model)
    return new_model


def reset_weights_to_checkpoint(model, ckp, skip_keyword=None):
    """Resets inplace. Skips if `skip_keyboard in weight.name`."""

    temp = tf.keras.models.clone_model(model)
    temp.load_weights(ckp)
    for w1, w2 in zip(model.weights, temp.weights):
        if skip_keyword in w1.name:
            continue
        w1.assign(w2)


def clip_many(values, clip_at, clip_from=None, inplace=False):
    """Clips a list of tf or np arrays. Returns tf arrays."""

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
    return np.concatenate([x.flatten() if isinstance(x, np.ndarray)
                           else x.numpy().flatten() for x in arrays], axis=0)


def visualize_masks(masks, mask_activation=None):
    n_plots = 5 if mask_activation else 3
    figheight = 20 if mask_activation else 14
    fig, axes = plt.subplots(n_plots, 1,
                             figsize=(7, figheight),
                             constrained_layout=True)
    masks = [mask.numpy() for mask in masks]
    if mask_activation:
        activated_masks = [mask_activation(mask).numpy() for mask in masks]
        axes[0].set_title('early layers not activated')
        axes[0].hist(concatenate_flattened(masks[:4]), bins=30)
        axes[1].set_title('early layers activated')
        axes[1].hist(concatenate_flattened(activated_masks[:4]), bins=30)
        axes[2].set_title('all layers not activated')
        axes[2].hist(concatenate_flattened(masks), bins=30)
        axes[3].set_title('all layers activated')
        axes[3].hist(concatenate_flattened(activated_masks), bins=30)
        averages = [np.mean(np.abs(mask)) for mask in activated_masks]
        last_plot = 4
    else:
        axes[0].set_title('early layers')
        axes[0].hist(concatenate_flattened(masks[:4]), bins=30)
        axes[1].set_title('all layers')
        axes[1].hist(concatenate_flattened(masks), bins=30)
        averages = [np.mean(np.abs(mask)) for mask in masks]
        last_plot = 2

    axes[last_plot].set_title('density of layers')
    axes[last_plot].bar(range(len(averages)), averages)
    fig.show()
    return fig


def log_mask_info(kernel_masks, mask_activation, logger=None):
    masks = [mask_activation(m) for m in kernel_masks]
    masks = np.abs(concatenate_flattened(masks))
    if logger:
        logger['avg_mask'] = np.mean(masks)
        logger['mask_std'] = np.std(masks)
    return masks


def compare_masks(perf_m, m, mask_activation, force_sparsity=None):
    from sklearn.metrics import precision_recall_curve
    m = np.abs(concatenate_flattened([mask_activation(x) for x in m]))
    perf_m = concatenate_flattened(perf_m)

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


def create_masked_layers(mask_activation):
    """With custom mask_activation."""

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


# %%

@tf.function
def train_step(x, y,
               model,
               optimizer,
               loss_fn,
               trainable_weights,
               variables_to_watch=None,
               logger=None):
    with tf.GradientTape() as tape:
        if variables_to_watch:
            tape.watch(variables_to_watch)
        outs = model(x, training=True)
        outs = tf.cast(outs, tf.float32)
        loss = loss_fn(y, outs)
        scaled_loss = optimizer.get_scaled_loss(loss)

    scaled_grads = tape.gradient(scaled_loss, trainable_weights)
    grads = optimizer.get_unscaled_gradients(scaled_grads)
    optimizer.apply_gradients(zip(grads, trainable_weights))
    acc = tf.keras.metrics.sparse_categorical_accuracy(y, outs)
    if logger:
        logger['train_loss'](loss)
        logger['train_acc'](acc)
    else:
        return loss, acc


@tf.function
def valid_step(x, y,
               model,
               loss_fn,
               logger=None,
               training=False):
    outs = model(x, training=training)
    outs = tf.cast(outs, tf.float32)
    loss = loss_fn(y, outs)
    acc = tf.keras.metrics.sparse_categorical_accuracy(y, outs)
    if logger:
        logger['valid_loss'](loss)
        logger['valid_acc'](acc)
    else:
        return loss, acc

# %%

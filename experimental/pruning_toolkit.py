import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf



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

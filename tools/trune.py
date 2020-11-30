import tensorflow as tf
import numpy as np

from itertools import islice

from tools.pruning import apply_pruning_for_model

from tensorflow.keras.mixed_precision import experimental as mixed_precision


def truning(model,
            learning_rate,
            momentum,
            weight_decay,
            num_iterations,
            steps_per_epoch,
            dataset):
    print("TRUNE TRAINING...")
    assert isinstance(model, tf.keras.Model)
    optimizer = tf.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

    kernel_masks = [w for w in model.weights if 'kernel_mask' in w.name]
    bernoulli_distribs = [tf.Variable(tf.zeros_like(mask),
                                      dtype=mask.dtype,
                                      trainable=True)
                          for mask in kernel_masks]
    for mask in kernel_masks:
        mask._trainable = True

    def take_from_dataset(ds, num_samples):
        return ds.take(num_samples), ds.skip(num_samples)

    def get_density(model):
        nonzero = 0
        max_nonzero = 0
        for layer in model.layers:
            if hasattr(layer, 'kernel_mask'):
                km = layer.kernel_mask.numpy()
                max_nonzero += km.size
                nonzero += km.sum()
        return nonzero / max_nonzero

    def regularize_kernel_mask(layer):
        def _f():
            return tf.reduce_sum(layer.kernel_mask)

        return _f

    regularization_losses = []
    for layer in model.layers:
        if hasattr(layer, 'kernel_mask'):
            regularization_losses.append(regularize_kernel_mask(layer))

    acc_metric = tf.metrics.SparseCategoricalAccuracy()
    loss_metric = tf.metrics.SparseCategoricalCrossentropy(True)

    @tf.function
    def train_step(x, y, decay):
        for kmask, distrib in zip(kernel_masks, bernoulli_distribs):
            clipped_mask = tf.sigmoid(distrib)
            binary_mask = tf.random.uniform(shape=clipped_mask.shape)
            kmask.assign(tf.cast(binary_mask < clipped_mask, kmask.dtype))

        with tf.GradientTape() as tape:
            outs = model(x)
            loss = loss_fn(y, outs)
            loss += tf.add_n([l() for l in regularization_losses]) * decay

        grads = tape.gradient(loss, kernel_masks)
        optimizer.apply_gradients(zip(grads, bernoulli_distribs))
        acc_metric(y, outs)
        loss_metric(y, outs)

        for mask in bernoulli_distribs:
            mask.assign(tf.clip_by_value(mask, -15, 15))

    @tf.function
    def train_epoch(num_steps, decay):
        for x, y in dataset.train.take(num_steps):
            train_step(x, y, decay)

    @tf.function
    def valid_epoch():
        for x, y in dataset.valid:
            outs = model(x, training=False)
            acc_metric(y, outs)
            loss_metric(y, outs)

    def reset_metrics():
        acc = acc_metric.result()
        loss = loss_metric.result()
        acc_metric.reset_states()
        loss_metric.reset_states()
        return acc, loss

    def set_expected_masks():
        for kmask, distrib in zip(kernel_masks, bernoulli_distribs):
            clipped_mask = tf.sigmoid(distrib)
            kmask.assign(tf.cast(0.5 < clipped_mask, kmask.dtype))

    decay = tf.Variable(weight_decay, trainable=False)

    steps_per_epoch = min(num_iterations, steps_per_epoch)
    for ep in range(int(num_iterations / steps_per_epoch)):
        train_epoch(steps_per_epoch, decay)
        tacc, tloss = reset_metrics()

        # set_expected_masks()
        # apply_pruning_for_model(model)

        valid_epoch()
        vacc, vloss = reset_metrics()
        density = get_density(model)

        print(f"EP {ep + 1}",
              f"DENSITY {density:6.4f}",
              f"VACC {vacc:6.4f}",
              f"TACC {tacc:6.4f}", sep=' | ')
    return model

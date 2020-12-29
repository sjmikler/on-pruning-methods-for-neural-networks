import tensorflow as tf
from tools.pruning import apply_pruning_for_model, report_density


def truning(model,
            learning_rate,
            momentum,
            weight_decay,
            num_iterations,
            steps_per_epoch,
            dataset):
    print("TRUNE TRAINING...")
    assert isinstance(model, tf.keras.Model)
    optimizer = tf.optimizers.SGD(learning_rate=learning_rate,
                                  momentum=momentum, nesterov=True)
    loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

    kernel_masks = [w for w in model.weights if "kernel_mask" in w.name]
    bernoulli_distribs = [tf.Variable(mask * 25 - 15) for mask in kernel_masks]

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
            binary_mask = tf.random.uniform(shape=clipped_mask.shape)
            kmask.assign(tf.cast(binary_mask < clipped_mask, kmask.dtype))

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
        optimizer.apply_gradients(zip(updates, bernoulli_distribs))

        for mask in bernoulli_distribs:
            mask.assign(tf.clip_by_value(mask, -15, 15))

    def train_epoch(ds, decay, num_iter):
        progbar = tf.keras.utils.Progbar(num_iter)

        for x, y in ds.take(num_iter):
            train_step(x, y, decay)
            progbar.add(1)

    @tf.function
    def valid_epoch(ds=dataset['valid']):
        for x, y in ds:
            outs = model(x, training=False)
            acc_metric(y, outs)
            loss_metric(y, outs)

    def reset_metrics():
        acc = acc_metric.result()
        loss = loss_metric.result()
        acc_metric.reset_states()
        loss_metric.reset_states()
        return acc, loss

    decay = tf.Variable(weight_decay, trainable=False)

    valid_epoch()
    vacc, vloss = reset_metrics()
    density = report_density(model)
    apply_pruning_for_model(model)

    print(f"EP {0}", f"DENSITY {density:6.4f}", f"VACC {vacc:6.4f}")

    num_iter = steps_per_epoch
    for ep in range(num_iterations // steps_per_epoch):
        train_epoch(dataset['train'], decay, num_iter)
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
            sep=" | ",
        )
    return model

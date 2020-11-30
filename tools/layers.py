import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


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

        self.sparsity = 1 - np.mean(self.kernel_mask.numpy())

    def call(self, x):
        masked_w = tf.multiply(self.kernel, self.kernel_mask)
        # masked_w = masked_w / tf.reduce_mean(self.kernel_mask)

        result = tf.matmul(x, masked_w)

        if self.use_bias:
            result = tf.add(result, self.bias)

        return self.activation(result)

    def set_pruning_mask(self, new_mask: np.ndarray):
        """
        :param new_mask: mask of the same shape as `layer.kernel`
        :return: None
        """
        tf.assert_equal(new_mask.shape, self.kernel_mask.shape)
        self.kernel_mask.assign(new_mask)
        self.sparsity = 1 - np.mean(self.kernel_mask.numpy())
        self.left_unpruned = np.sum(self.kernel_mask.numpy() == 1)

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
        self.sparsity = 1 - np.mean(self.kernel_mask.numpy())

    def call(self, x):
        masked_w = tf.multiply(self.kernel, self.kernel_mask)
        # masked_w = masked_w / tf.reduce_mean(self.kernel_mask)

        result = tf.nn.conv2d(
            x, masked_w, strides=self.strides, padding=self.padding.upper()
        )

        if self.use_bias:
            result = tf.add(result, self.bias)

        return self.activation(result)

    def set_pruning_mask(self, new_mask: np.ndarray):
        """
        :param new_mask: mask of the same shape as `layer.kernel`
        :return: None
        """
        tf.assert_equal(new_mask.shape, self.kernel_mask.shape)
        self.kernel_mask.assign(new_mask)
        self.sparsity = 1 - np.mean(self.kernel_mask.numpy())
        self.left_unpruned = np.sum(self.kernel_mask.numpy() == 1)

    def apply_pruning_mask(self):
        self.kernel.assign(tf.multiply(self.kernel, self.kernel_mask))


class GumbelMaskedConv(tf.keras.layers.Conv2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gumbel = tfp.distributions.Gumbel(0, 1)

    def build(self, input_shape):
        super().build(input_shape)

        self.d0 = self.add_weight(
            name="d0",
            shape=self.kernel.shape,
            dtype=self.kernel.dtype,
            initializer="ones",
            trainable=False,
        )
        self.d1 = self.add_weight(
            name="d1",
            shape=self.kernel.shape,
            dtype=self.kernel.dtype,
            initializer="ones",
            trainable=False,
        )
        self.tau = tf.Variable(1, dtype=tf.float32, name='tau')

    def get_soft_sample(self):
        v0 = tf.exp((self.d0 + self.gumbel.sample(self.d0.shape)) / self.tau)
        v1 = tf.exp((self.d1 + self.gumbel.sample(self.d1.shape)) / self.tau)
        soft_sample = v1 / (v0 + v1)
        return soft_sample

    def call(self, x):
        soft_sample = self.get_soft_sample()

        masked_w = tf.multiply(self.kernel, soft_sample)
        masked_w = masked_w / tf.reduce_mean(soft_sample)
        result = tf.nn.conv2d(
            x, masked_w, strides=self.strides, padding=self.padding.upper()
        )

        if self.use_bias:
            result = tf.add(result, self.bias)

        return self.activation(result)


class GumbelMaskedDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gumbel = tfp.distributions.Gumbel(0, 1)

    def build(self, input_shape):
        super().build(input_shape)

        self.d0 = self.add_weight(
            name="d0",
            shape=self.kernel.shape,
            dtype=self.kernel.dtype,
            initializer="ones",
            trainable=False,
        )
        self.d1 = self.add_weight(
            name="d1",
            shape=self.kernel.shape,
            dtype=self.kernel.dtype,
            initializer="ones",
            trainable=False,
        )
        self.tau = tf.Variable(1, dtype=tf.float32, name='tau')

    def get_soft_sample(self):
        v0 = tf.exp((self.d0 + self.gumbel.sample(self.d0.shape)) / self.tau)
        v1 = tf.exp((self.d1 + self.gumbel.sample(self.d1.shape)) / self.tau)
        soft_sample = v1 / (v0 + v1)
        return soft_sample

    def call(self, x):
        soft_sample = self.get_soft_sample()
        masked_w = tf.multiply(self.kernel, soft_sample)
        masked_w = masked_w / tf.reduce_mean(soft_sample)
        result = tf.matmul(x, masked_w)

        if self.use_bias:
            result = tf.add(result, self.bias)

        return self.activation(result)

import tensorflow as tf
import tensorflow_datasets as tfds


def cifar(train_batch_size=128,
          valid_batch_size=128,
          padding='reflect',
          dtype=tf.float32,
          shuffle_train=20000,
          repeat_train=True,
          version=10):
    subtract = tf.constant([0.49139968, 0.48215841, 0.44653091], dtype=dtype)
    divide = tf.constant([0.24703223, 0.24348513, 0.26158784], dtype=dtype)

    def train_prep(x, y):
        x = tf.cast(x, dtype) / 255.0
        x = tf.image.random_flip_left_right(x)
        x = tf.pad(x, [[4, 4], [4, 4], [0, 0]], mode=padding)
        x = tf.image.random_crop(x, (32, 32, 3))
        x = (x - subtract) / divide
        return x, y

    def valid_prep(x, y):
        x = tf.cast(x, dtype) / 255.0
        x = (x - subtract) / divide
        return x, y

    if version == 10:
        ds = tfds.load(name='cifar10', as_supervised=True)
    elif version == 100:
        ds = tfds.load(name='cifar100', as_supervised=True)
    else:
        raise exception(f"version = {version}, but should be from (10, 100)!")

    if repeat_train:
        ds['train'] = ds['train'].repeat()
    if shuffle_train:
        ds['train'] = ds['train'].shuffle(shuffle_train)
    ds['train'] = ds['train'].map(train_prep)
    ds['train'] = ds['train'].batch(train_batch_size)

    ds['valid'] = ds.pop('test')
    ds['valid'] = ds['valid'].map(valid_prep)
    ds['valid'] = ds['valid'].batch(valid_batch_size)
    return ds


def mnist(train_batch_size=100,
          valid_batch_size=100,
          dtype=tf.float32,
          shuffle_train=10000):
    def preprocess(x, y):
        x = tf.cast(x, dtype)
        x /= 255
        return x, y

    ds = tfds.load(name='mnist', as_supervised=True)
    ds['train'] = ds['train'].repeat()
    ds['train'] = ds['train'].shuffle(shuffle_train)
    ds['train'] = ds['train'].map(preprocess)
    ds['train'] = ds['train'].batch(train_batch_size)

    ds['valid'] = ds.pop('test')
    ds['valid'] = ds['valid'].map(preprocess)
    ds['valid'] = ds['valid'].batch(valid_batch_size)
    return ds


def test(train_batch_size=100,
         image_shape=(32, 32, 3),
         dtype=tf.float32):
    images = tf.ones([2, *image_shape])
    target = tf.constant([0, 1])

    def preprocess(x, y):
        x = tf.cast(x, dtype)
        return x, y

    ds = {}
    ds['train'] = tf.data.Dataset.from_tensor_slices((images, target))
    ds['train'] = ds['train'].map(preprocess).repeat().batch(train_batch_size)
    ds['valid'] = tf.data.Dataset.from_tensor_slices((images, target))
    ds['valid'] = ds['valid'].map(preprocess).batch(2)
    return ds

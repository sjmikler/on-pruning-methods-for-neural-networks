import tensorflow as tf
import tensorflow_datasets as tfds

from tools.utils import ddict


def cifar10(train_batch_size=128,
            valid_batch_size=128,
            padding='reflect',
            dtype=tf.float32,
            shuffle_train=10000,
            repeat_train=True):
    subtract = [0.49139968, 0.48215841, 0.44653091]
    divide = [0.24703223, 0.24348513, 0.26158784]

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

    ds = tfds.load(name='cifar10', as_supervised=True)
    ds = ddict(ds)
    if repeat_train:
        ds.train = ds.train.repeat()
    if shuffle_train:
        ds.train = ds.train.shuffle(shuffle_train)
    ds.train = ds.train.map(train_prep)
    ds.train = ds.train.batch(train_batch_size)

    ds.valid = ds.pop('test')
    ds.valid = ds.valid.map(valid_prep)
    ds.valid = ds.valid.batch(valid_batch_size)
    return ds


def mnist(train_batch_size=100,
          valid_batch_size=100,
          dtype=tf.float32,
          shuffle=10000):
    def preprocess(x, y):
        x = tf.cast(x, dtype)
        x /= 255
        return x, y

    ds = tfds.load(name='mnist', as_supervised=True)
    ds = ddict(ds)
    ds.train = ds.train.repeat()
    ds.train = ds.train.shuffle(shuffle)
    ds.train = ds.train.map(preprocess)
    ds.train = ds.train.batch(train_batch_size)

    ds.valid = ds.pop('test')
    ds.valid = ds.valid.map(preprocess)
    ds.valid = ds.valid.batch(valid_batch_size)
    return ds


def get_dataset(ds_name, precision, **config):
    if ds_name == 'cifar10':
        return cifar10(dtype=tf.float16 if precision == 16 else tf.float32,
                       shuffle_train=20000,
                       **config)
    elif ds_name == 'mnist':
        return mnist(dtype=tf.float16 if precision == 16 else tf.float32,
                     shuffle=5000,
                     **config)
    else:
        raise KeyError(f"DATASET {ds_name} NOT RECOGNIZED!")

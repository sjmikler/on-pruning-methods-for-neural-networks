import tensorflow as tf
import tensorflow_datasets as tfds

from tools.utils import ddict


def cifar(train_batch_size=128,
          valid_batch_size=128,
          padding='reflect',
          dtype=tf.float32,
          shuffle_train=20000,
          repeat_train=True,
          version=10):
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

    if version == 10:
        ds = tfds.load(name='cifar10', as_supervised=True)
    elif version == 100:
        ds = tfds.load(name='cifar100', as_supervised=True)
    else:
        raise exception(f"version = {version}, but should be from (10, 100)!")

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
          shuffle_train=10000):
    def preprocess(x, y):
        x = tf.cast(x, dtype)
        x /= 255
        return x, y

    ds = tfds.load(name='mnist', as_supervised=True)
    ds = ddict(ds)
    ds.train = ds.train.repeat()
    ds.train = ds.train.shuffle(shuffle_train)
    ds.train = ds.train.map(preprocess)
    ds.train = ds.train.batch(train_batch_size)

    ds.valid = ds.pop('test')
    ds.valid = ds.valid.map(preprocess)
    ds.valid = ds.valid.batch(valid_batch_size)
    return ds


def get_dataset(ds_name, precision, **config):
    if ds_name == 'cifar10':
        return cifar(dtype=tf.float16 if precision == 16 else tf.float32,
                     version=10,
                     **config)
    if ds_name == 'cifar100':
        return cifar(dtype=tf.float16 if precision == 16 else tf.float32,
                     version=100,
                     **config)
    elif ds_name == 'mnist':
        return mnist(dtype=tf.float16 if precision == 16 else tf.float32,
                     **config)
    else:
        raise KeyError(f"DATASET {ds_name} NOT RECOGNIZED!")

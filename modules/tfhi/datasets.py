import tensorflow as tf
import tensorflow_datasets as tfds

try:
    from ._initialize import *
except ImportError:
    pass


def postprocess_tf_dataset(
    ds,
    train_bs,
    valid_bs,
    train_map=None,
    valid_map=None,
    train_shuffle=1000,
    train_repeat=True,
    prefetch=False,
    train="train",
    valid="test",
):
    if train_repeat:
        ds[train] = ds[train].repeat()
    if train_shuffle:
        ds[train] = ds[train].shuffle(train_shuffle)
    if train_map:
        ds[train] = ds[train].map(train_map)
    ds[train] = ds[train].batch(train_bs)

    ds[valid] = ds[valid].map(valid_map)
    ds[valid] = ds[valid].batch(valid_bs)

    if prefetch:
        ds[train] = ds[train].prefetch(tf.data.experimental.AUTOTUNE)
        ds[valid] = ds[valid].prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def cifar(
    padding="reflect", dtype=tf.float32, version=10, data_dir=None, **kwds,
):
    if version == 10 or version == 100:
        ds = tfds.load(name=f"cifar{version}", as_supervised=True, data_dir=data_dir)
    else:
        raise Exception(f"version = {version}, but should be either 10 or 100!")

    subtract = tf.constant([0.49139968, 0.48215841, 0.44653091], dtype=dtype)
    divide = tf.constant([0.24703223, 0.24348513, 0.26158784], dtype=dtype)

    def train_map(x, y):
        x = tf.cast(x, dtype) / 255.0
        x = tf.image.random_flip_left_right(x)
        x = tf.pad(x, [[4, 4], [4, 4], [0, 0]], mode=padding)
        x = tf.image.random_crop(x, (32, 32, 3))
        x = (x - subtract) / divide
        return x, y

    def valid_map(x, y):
        x = tf.cast(x, dtype) / 255.0
        x = (x - subtract) / divide
        return x, y

    kwds.setdefault("train_bs", 128)
    kwds.setdefault("valid_bs", 512)
    kwds.setdefault("train_shuffle", 20000)
    return postprocess_tf_dataset(ds, train_map=train_map, valid_map=valid_map, **kwds)


def mnist(dtype=tf.float32, data_dir=None, **kwds):
    ds = tfds.load(name="mnist", as_supervised=True, data_dir=data_dir)

    def preprocess(x, y):
        x = tf.cast(x, dtype)
        x /= 255
        return x, y

    kwds.setdefault("train_bs", 100)
    kwds.setdefault("valid_bs", 500)
    kwds.setdefault("train_shuffle", 10000)
    return postprocess_tf_dataset(
        ds, train_map=preprocess, valid_map=preprocess, **kwds
    )


def test(train_batch_size=100, image_shape=(32, 32, 3), dtype=tf.float32):
    images = tf.ones([2, *image_shape])
    target = tf.constant([0, 1])

    def preprocess(x, y):
        x = tf.cast(x, dtype)
        return x, y

    ds = dict()
    ds["train"] = tf.data.Dataset.from_tensor_slices((images, target))
    ds["train"] = ds["train"].map(preprocess).repeat().batch(train_batch_size)
    ds["test"] = tf.data.Dataset.from_tensor_slices((images, target))
    ds["test"] = ds["test"].map(preprocess).batch(2)
    return ds


def get_dataset_from_alias(alias, precision=32):
    assert isinstance(alias, str)

    if precision == 16:
        dtype = tf.float16
    elif precision == 32:
        dtype = tf.float32
    elif precision == 64:
        dtype = tf.float64
    else:
        raise NotImplementedError(f"Unknown precision {precision}!")

    if alias == "cifar10":
        return cifar(dtype=dtype, version=10)
    elif alias == "cifar100":
        return cifar(dtype=dtype, version=100)
    elif alias == "mnist":
        return mnist(dtype=dtype)
    else:
        raise NotImplementedError(f"Unknown alias {alias}")


def figure_out_input_shape(ds, train="train", valid="test"):
    for x, y in ds[valid]:
        break
    else:
        raise RuntimeError("Dataset is empty!")
    return x.shape[1:]


def figure_out_n_classes(ds, train="train", valid="test"):
    classes = set()
    for x, y in ds[valid]:
        classes.update(y.numpy())
    return len(classes)

import tensorflow as tf

try:
    from ._initialize import *
except ImportError:
    pass


def log_from_history(history, exp):
    import datetime

    min_loss = min(history["val_loss"])
    max_acc = max(history["val_accuracy"])
    final_acc = history["val_accuracy"][-1]

    min_tr_loss = min(history["loss"])
    max_tr_acc = max(history["accuracy"])

    print(f"BEST ACCURACY: {max_acc}")

    exp.TIME = datetime.datetime.now().strftime("%Y.%m.%d %H:%M")
    exp.ACC = max_acc
    exp.FINAL_ACCU = final_acc
    exp.VALID_LOSS = min_loss
    exp.TRAIN_ACCU = max_tr_acc
    exp.TRAIN_LOSS = min_tr_loss

    if hasattr(exp, "tensorboard_log") and exp.tensorboard_log:
        writer = tf.summary.create_file_writer(exp.tensorboard_log)
        with writer.as_default():
            for key in history:
                for idx, value in enumerate(history[key]):
                    tf.summary.scalar(key, value, idx + 1)
            tf.summary.text("experiment", data=str(exp), step=0)
    return exp

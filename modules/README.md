# tf-helper

**Available command line arguments**

```
optional arguments:
  -h, --help          show this help message and exit
  --gpu GPU           Which GPUs to use during training, e.g. 0,1,3 or 1
  --no-memory-growth  Disables memory growth
```

**Available experiment parameters**

* `precision` is either 16, 32 or 64

# pruning

Inherits from: **tf-helper**. You can use arguments and parameters from there.

**Available experiment parameters**

* `checkpointBP` is **checkpoint Before Pruning**. You can load full checkpoint here.
* `checkpointAP` is **checkpoint After Pruning**. Kernel masks from the checkpoint will be skipped. This is used for some pruning methods.
* `pruning` and `pruning_config` settings
* `dataset` and `dataset_config` settings
* `model` and `model_config` settings
* `tensorboard_log`
* `weight_checkpoint`

**Fun Facts**

* Tensorboard logs with training and validation history are saved all at once, after the training in `experiment.yaml/tensorboard_log`. Nothing will be saved if training is interrupted.

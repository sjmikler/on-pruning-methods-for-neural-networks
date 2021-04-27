# tf-helper

**Available command line arguments**

```
optional arguments:
  -h, --help          show this help message and exit
  --gpu GPU           Which GPUs to use during training, e.g. 0,1,3 or 1
  --no-memory-growth  Disables memory growth
```

**Available experiment parameters**

* `precision` is 16 for mixed precision or anything else to skipping

# pruning

Inherits from: **tf-helper**. You can use arguments and parameters from there.

**Available experiment parameters**

* `checkpointAP` is **checkpoint After Pruning**. You can load almost full checkpoint here: kernel masks from the checkpoints will be skipped. This allows for a few pruning techniques.
* `checkpointBP` is **checkpoint Before Pruning**. You can load full checkpoint here.
* `pruning` and `pruning_config` settings

**Fun Facts**

* Tensorboard logs with training and validation history are saved all at once, **after** the training in `experiment.yaml/full_path`. Nothing will be saved if training is interrupted.

# General

Use those names if applicable for a module:

* `steps` for total number of steps in the training
* `steps_per_epoch`
* `model` and `model_config`
* `dataset` and `dataset_config`
* `optimizer` and `optimizer_config`

# Pruning

**Available parameters**

* `checkpointAP` is **checkpoint After Pruning**
* `checkpointBP` is **checkpoint Before Pruning**. You can load full checkpoint before pruning, but after pruning **pruning masks from the checkpoint will be skipped**. This allows for a few pruning techniques
* `pruning` and `pruning_config`

**Facts**

* Tensorboard logs with training and validation history are saved all at once, **after** the training in `experiment.yaml/full_path`. 

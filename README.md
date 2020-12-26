### Experiment definition: `experiments.yaml`

Contains experiments. The first element is **global config** which contains default values for experiments. All the
experiments will be updated with it, but values specified later, in experiments, have priority over **global config**.

**Special names**

1. `REPEAT`: copies an experiment many times before fancy parsing, can be used for iterative trainings
2. `GLOBAL_REPEAT`: performs all listed experiments many times (can be used to repeat iterative trainings)
3. `REP`: is added by `run.py` and is a repetition index in range `[0, REPEAT-1]`.
4. `RND_IDX`: is added by `run.py` and can be used to uniquely identify an experiment.
5. `checkpoint` will be used by `run.py` for `model.save_weights(ckp)` method.

### Script `run.py`

Runs trainings specified in `experiment.yaml`. If `queue` parameter is specified as a valid path, the queue of the
experiments will be stored in it and can be modified when experiments are running. Otherwise, queue is stored in RAM
memory and cannot be modified.

1. You can use `--dry` flag to test your experiment parsing
2. You can use any flag, like `--sparsity=0.5` or `--precision=32` to update **global config** from command line.
3. `steps_per_epoch` should be larger than number of batches in the dataset. Otherwise, you will not use all the samples
   during the training.
4. `checkpointBP` is **checkpoint After Pruning** and `checkpointBP` is **checkpoint Before Pruning**. You can load any
   checkpoint before pruning, but after pruning **masks from the checkpoint will be skipped**.
5. If experiment is stopped with `KeyboardInterrupt`, there will be 2 second pause during which `run.py` can be
   interrupted completely. If not interrupted completely, next experiment in the queue will start instead. Interrupted
   experiments will not leave any checkpoints.
6. If `name: skip`, training will not be performed, but values (like `sparsity: 0.0`) can be used in fancy parsing.
   Skipped experiments will not leave any checkpoints.

```
PROCEDURES IN ORDER:
1. Creating model
2. Loading checkpoint Before Pruning
3. Applying pruning
4. Loading checkpoint After Pruning (skip masks from checkpoint)
5. Pruning related procedures After Pruning (like shuffling masks)
```

Before running the training, experiments will be parsed...

### Fancy experiment parsing

Values for experiments can be specified explicitly, e.g. `sparsity: 0.9`, but there are some tricks to simplify longer
and more complicated experiments.

* You can use `parameter[n]` to reuse value of parameter named `parameter` from experiment number `n`.

* You can use `exec` in the beginning of a parameter value to execute the statement in Python and do super cool
  tricks...

#### Tricks:

1. `sparsity: exec 0.5 * sparsity[-1]` will be parsed as half of the sparsity of previous experiment in the queue.

2. `checkpoint: exec '\'.join([directory, name])` will work as in Python, with variable scope from current experiment.

3. When `parameter[idx]` is used in nested dict:

```
...
pruning_config:
   sparsity: 0.5
---
sparsity: sparsity[0] # THIS WON'T PARSE (will raise KeyError)
pruning_config:
   sparsity: sparsity[0] # THIS WILL PARSE
```

4. Parsing order:

```
...
name: test
---
model: VGG
name: test2
directory: exec f"{name}/{name[-1]}/{model}
```

> You can do this, because `name[-1]` will be reduced to `test` before executing `exec` statement. This means the following:

```
...
list: [1, 2, 3]
---
list: [2, 3, 4]
number: exec list[0]
```

> `number` will be `[1, 2, 3]`, not `2`.

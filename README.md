## Defining experiments

File `experiments.yaml` defines experiments. Its first element is always **global
config** which contains default values for experiments. If a settings repeats between **
default config** and experiments, experiments have priority.

**Special names**

1. `REPEAT`: copies a single experiment many times before fancy parsing, can be used for
   iterative trainings
2. `GLOBAL_REPEAT`: performs all listed experiments many times (can be used to repeat
   iterative trainings)
3. `REP`: is added by `run.py` and is a repetition index in range `[0, REPEAT-1]`
4. `RND_IDX`: is added by `run.py` and can be used to uniquely identify an experiment
5. `full_path`: location of `tensorboard` logs
6. `checkpoint`: is used by `run.py` for `model.save_weights(ckp)` method
7. `yaml_logdir`: location of shorter `.yaml` logs

## Running experiments

Script `run.py` runs trainings specified in `experiment.yaml`. If `queue` parameter is
specified as a valid path, the queue of the experiments will be stored on it as
a `.yaml` file and can be modified when experiments are running. Otherwise, queue is
stored in RAM memory and cannot be modified.

1. You can use `--dry` flag without any arguments to test your experiment parsing
2. You can use any flag, like `--sparsity=0.5` or `--precision=32` to update **global
   config** straight from command line. Mainly intended for hardware settings,
   like `--memory-growth=False`
3. `steps_per_epoch` should be larger than number of batches in the dataset. Otherwise,
   you will not use all the samples during the training
4. `checkpointAP` is **checkpoint After Pruning** and `checkpointBP` is **checkpoint
   Before Pruning**. You can load full checkpoint before pruning, but after pruning **
   pruning masks from the checkpoint will be skipped**. This allows for many pruning
   techniques
5. If experiment is stopped with `KeyboardInterrupt`, there will be 2 second pause
   during which `run.py` can be interrupted completely. If not interrupted completely,
   next experiment in the queue will start instead. Interrupted experiments will not
   leave any checkpoints
6. If `name: skip`, training will not be performed, but values (like `sparsity: 0.0`)
   can be used in fancy parsing. Skipped experiments will not leave any checkpoints

```
PROCEDURES IN ORDER:
1. Creating model
2. Loading checkpoint Before Pruning
3. Applying pruning
4. Loading checkpoint After Pruning (skip masks from checkpoint)
5. Pruning related procedures After Pruning (like shuffling masks)
```

Before running the training, experiments will be parsed...

## Fancy parsing with `eval`

> **WARNING**: `eval` is considered unsafe, be sure only you accessed your `experiment.yaml`

All values for experiments can be specified explicitly, e.g. `sparsity: 0.9`, but there
are some tricks to simplify longer and more complicated experiment procedures.

Fancy parsing allows you to execute python code during parsing. To do this, you need to
type `eval` in the beginning of a parameter value. With this, you can do super cool
tricks...

#### Tricks:

1. `odd_name: eval '\'.join([directory, name])` will work as in Python, because eval
   uses variable scope from current experiment and executes the code using `eval`
   function. Will only work if both `directory` and `name` are specified in current
   experiment before `odd_name` or are specified in default config.

2. `sparsity: eval 0.5 * E[-1].sparsity` will be parsed as half of the sparsity of
   previous experiment in the queue. Before running `eval`, list `E` is added to the
   scope which allows access to previous experiments.

3. You can access values from nested dictionaries:

```
...
pruning_config:
   odd_config:
      sparsity: 0.5
---
sparsity: eval E[-1].pruning_config.odd_config.sparsity
pruning_config:
   sparsity: eval E[-1].pruning_config.odd_config.sparsity
```

4. Other examples:

```
...
name: test
---
model: VGG
name: test2
directory: eval f"{name}/{name[-1]}/{model}"
```

> Resulting `directory` value will be `test2/test/VGG`

```
...
list: [1, 2, 3]
---
list: [2, 3, 4]
number2: eval list[0]
number3: eval E[-1].list[2]
```

> Resulting `number2` value will be `2` and `number3` value will be `3`.

```
...
nested:
   test: 1

incremented: eval nested.test + 1
```

> Resulting `incremented` value will be `2`.

5. What won't work

```
...
list1: [2, 3, 4]
test: eval list1 # this will work

nested:
   list2: [2, 3, 4]
   test1: eval list2 # this will work as list2 is on the same level
   test: eval list1 # this won't work as list1 is one level higher
```

This is because nested dicts have access only to variables on their level or deeper. If
you run the code, you will see `NameError: name list1 is not defined`.

## Logs management

Logs in `.yaml` format are saved in location passed as `experiment.yaml/yaml_logdir`.
These contain experiment formulation and best test accuracy of a network. Following tool
can recursively collect logs from `.yaml` files in subdirectories of `path`.

```
collect_logs.py 
   --path=[will be recursively serached for .yaml logs] 
   --dest=[where cumulative log will be saved, end it with .yaml]
```

Examples:

```
python -m tools.collect_logs.py 
   --path=data/VGG19_IMP03_ticket 
   --dest=data/VGG19_IMP03_ticket/collected_logs.yaml
```

```
python -m tools.collect_logs.py 
   -p=data/VGG19_IMP03_ticket 
   -d=collected_logs.yaml
```

```
python -m tools.collect_logs.py 
   -p=data/VGG19_IMP03_ticket 
```

By default, `--dest` sets itself to the same value as `--path`.

Tensorboard logs with training and validation history are saved all at once, **after**
the training in `experiment.yaml/full_path`. 

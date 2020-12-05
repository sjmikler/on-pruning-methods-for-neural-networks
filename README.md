### Special files

* `experiments.yaml`

Contains experiments. The first element is **global config** which contains default values for experiments. All the
experiments will be updated with it, but values specified later, in experiments have priority over **global config**.

**Special names**

1. `REPEAT`: copies an experiment many times before fancy parsing, can be used for iterative trainings
2. `GLOBAL_REPEAT`: performs all listed experiments many times (can be used to repeat iterative trainings)
3. `IDX`: is added in `run.py` and is constructed like this: `{RANDOM_INDEX} / {REPETITION}`

* Script `run.py`

1. You can use `--dry` flag to test your experiment parsing
2. You can use any flag, like `--sparsity=0.5` or `--precision=32` to update **global config** from command line.

### Fancy experiment parsing

You can use `parameter[n]` to reuse value of parameter named `parameter` from experiment number `n`.

You can use `exec` in the beginning of a parameter value to execute the statement in Python and do super cool tricks:

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

You can do this, because `name[-1]` will be reduced to `test` before executing `exec` statement. This means the
following:

```
...
list: [1, 2, 3]
---
list: [2, 3, 4]
number: exec list[0]
```

`number` will be `[1, 2, 3]`, not `2`.

Experiments running orchestra that manages definitions and results of experiments.

```
Requirements:
- python>=3.8
- pyyaml>=5.1
```

# Defining experiments

There's a single file that contains your **experiment definition** (by default `experiment.yaml`). Its first element is always **default config** which contains default values for experiments that come after it. If a settings repeats between default config and experiments, experiments have the priority.

**Names added by `run.py`**

1. `REP`: repetition index in range `[0, REPEAT-1]`
2. `RND_IDX`: used to uniquely identify an experiment. Can be set manually
3. `TIME_ELAPSED`: time it took to run the module, in seconds

**Minimal experiment**

Following parameters are required by `run.py`:

```
GlobalRepeat: 1     # performs all listed experiments many times. Resulting order: `1, 2, 1, 2`
GlobalQueue: null   # if valid path, create an experiment queue available to modify on the hard drive
Repeat: 1           # copies a single experiment many times **before** fancy parsing. Resulting order: `1, 1, 2, 2`
Name: test          # if Name==skip, experiment will be skipped. Besides that, it is not used by `run.py`
YamlLog: log.yaml   # experiment definition with additional information (if added by a Module) will be saved there
Module: modules.example      # location of the module that will be run by run.py
```

There are conventions for parameter names:

* `parameter` is used in modules
* `Parameter` is used in `run.py` for experiment parsing and saving logs
* `PARAMETER` is added by scripts, not by user, and should be used to write results, e.g. `ACCURACY`

Custom modules will require custom parameters. It is advised to write flexible module with many parameters.

If in need, use following parameter names for consistency:

* `steps`: total number of steps in the training
* `steps_per_epoch`
* `model` with `model_config`
* `dataset` with `dataset_config`
* `optimizer` with `optimizer_config`
* `full_path`: location of `tensorboard` logs
* `checkpoint`: location of model checkpoints

# Running experiments

Script `run.py` launches experiments specified in **experiment definition**.

* If `Queue` parameter is specified as a valid path, the Queue of the experiments will be stored as a `.yaml` file and can be modified when experiments are running. Otherwise, Queue is stored in RAM memory and cannot be modified.


* If experiment is stopped with `KeyboardInterrupt`, there will be 2 second pause during which `run.py` can be interrupted completely by a second `KeyboardInterrupt`. If not interrupted completely, next experiment in the Queue will start.

* If `Name: skip`, training will not be performed, but experiment parameters can be used in fancy parsing. Skipped experiments will not launch the Module.

* `run.py` has its own command line arguments. Modules might contain their own command line arguments and to use them, you should pass them to `run.py` and it will pass them further down to the Module. If argument Name is used by both `run.py` and the Module, it will be used by both.
   ```
   > python run.py --help
   optional arguments:
     -h, --help            show this help message and exit
     --dry                 Skip execution but parse experiments
     --exp EXP             Path to .yaml file with experiments
     --pick PICK, --cherrypick-experiments PICK
                           Run only selected experiments, e.g. 0,1,3 or 1
   ```

* You can update **global config** straight from command line by passing arguments with `+` prefix instead of `-`, e.g. `python run.py +Queue Queue.yaml`.

# Modules

Module is a function that accepts one argument `exp` which is a dictionary with experiment details and is defined in **experiment definition** (e.g. `experiment.yaml`). They contain code for running an actual experiment. They use experiments parsed by `run.py`. Besides these parameters, they might define their own command line arguments. It is recommended that only a few hardware related settings are defined as command line arguments and the rest are in **experiment definition**.

[Available modules](modules/README.md)

# python run.py

`run.py` is parsing the experiments and launching a Module to do its work. Specific Module which `run.py` should be specified as a parameter `Module` in **default config** of **experiment definition**. Before Module starts the training, experiments will be parsed and there are a few tricks to it.

## Fancy parsing with `eval`

> **WARNING**: `eval` is considered unsafe - if someone has access to your **experiment definition**, they can run malicious code

All values for experiments can be specified explicitly in `experiment.yaml` as a number, string, dictionary or list, e.g. `sparsity: 0.9`, but there are some tricks that can simplify longer and complicated experiments. Fancy parsing allows you to execute python code during parsing. To do this, you need to type `eval` in the beginning of a parameter value. With this, you can do really cool tricks.

#### Tricks:

1. `odd_name: eval '\'.join([directory, Name])` will run real Python code and the result will be saved as `odd_name`. Code will be executed using `eval` function with variable scope from current experiment. In special case, if `directory` is fancy-parsed too, `directory` should be resolved before `odd_name`. **Default config** values can be fancy-parsed too, but they are resolved at the end.

2. `sparsity: eval 0.5 * E[-1].sparsity` will be parsed as half of the `sparsity` value from the previous experiment in the experiment Queue. This works, because before running `eval`, list `E` is added to the scope and this allows access to previous experiments.

3. You can access values from nested dictionaries using `.` or standard indexing:

```
pruning_config:
   odd_config:
      sparsity: 0.5
---
sparsity: eval E[-1].pruning_config.odd_config.sparsity
pruning_config:
   sparsity: eval E[-1]['pruning_config']['odd_config']['sparsity']
```

4. You can access values from parents' levels:

```
abc: 1
nested:
   abc: 2
   nested2:
      test: eval abc
```

This will work and will resolve to `test: 2`. Deeper levels have priority in this case.

5. Other examples:

```
Name: test
---
model: VGG
Name: test2
directory: eval f"{Name}/{E[-1].Name}/{model}"
```

> Resulting `directory` value will be `test2/test/VGG`.

```
list: [1, 2, 3]
---
list: [2, 3, 4]
number2: eval list[0]
number3: eval E[-1].list[2]
```

> Resulting `number2` value will be `2` and `number3` value will be `3`.

```
nested:
   test: 1

incremented: eval nested.test + 1
```

> Resulting `incremented` value will be `2`.

```
# DEFAULT CONFIG
test1: eval test3 - 1
---
test2: 5
test3: eval test2 - 1
```

> This is correct. Eval from **default config** is resolved at the end - `test3: 4`, `test1: 3`.

Following examples **won't work**

```
test3: eval test2 - 1  # ERROR: test2 IS UNKNOWN (WRONG ORDER)
test2: 5
```

```
# DEFAULT CONFIG
test1: 1
---
test2: eval test1 # ERROR: test1 IS UNKNOWN (CAN'T REFERENCE DEFAULT CONFIG)
```

Order is important for fancy parsing. If you need to reference a default value, redefine it in the experiment instead.

## Logs management

Modules are encouraged to leave meaningful logs in `.yaml` format. They can use location `experiment.yaml/yaml_logdir` for consistency. These logs should contain experiment formulation and basic information about the results, e.g. final accuracy. There are some tools that can make it easier to deal with large number of experiments.

### python collect.py

Recursively collects logs from `.yaml` files in subdirectories of `path`.

```
positional arguments:
  path                  starting directories for recursive log collecting

optional arguments:
  -h, --help            show this help message and exit
  --exclude [EXCLUDE [EXCLUDE ...]]
                        skip directories or files
  --dest DEST           directory of new .yaml file
  -v, --verbose         print visited directories during recursive collecting
```

**Examples:**

`python collect.py data/ --dest collected_logs --exclude unwanted_directory`

`python collect.py temp/ data/ -v` will collect logs from `temp` and `data` directories and will be verbose.

### python filter.py

Can be used to apply a query on `.yaml` log file. Available operations are listed in arguments. Arguments (query) can be read from a file using `$FILENAME`.

```
Filtering and sorting of .yaml logs. Use %FILENAME to load arguments from file.

positional arguments:
  path                  path to .yaml file containing logs

optional arguments:
  -h, --help            show this help message and exit
  --dest DEST           directory of new .yaml file
  --filter FILTER       python lambda function accepting experiment dict and returning boolean
  --sort SORT           python lambda function accepting experiment dict and returning sorting keys
  --reverse             reverse sorting order
  --keep-keys [KEEP_KEYS [KEEP_KEYS ...]]
                        keys in the experiment dict that should be kept, skip to keep everything
```

**Examples:**

`python filter.py logs_2021-01-01_12-00-00.yaml logs_2021-01-01_12-05-30.yaml %args`

where `args` is:

```
--filter=lambda exp: 'TIME' in exp
--filter=lambda exp: 'ACC' in exp and exp['ACC'] > 0.9
--sort=lambda exp: exp['TIME']
--sort=lambda exp: exp['Name']
--reverse
--keep-keys
ACC
RND_IDX
Name
model
TIME
```

As a result, all returned logs:

* contained `TIME` key
* contained `ACC` key with value larger than 0.9
* will be sorted by `TIME` values in reverse order
* then will be (stable) sorted by `Name` values in reverse order
* each experiment will contain only `ACC`, `RND_IDX`, `Name`, `model`, `TIME` keys

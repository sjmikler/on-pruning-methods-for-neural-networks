### Quick Start

```
python run.py --exp experiment-resnet.yaml
python run.py --exp experiment-vgg.yaml
python run.py --exp experiment-wrn.yaml
```

---

### Original repository: https://github.com/gahaalt/cool-runner

---

Experiments running orchestra that manages definitions and results of experiments.

```
Requirements:
- python>=3.8
- pyyaml>=5.1
- slackclient (not necessary)
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
Global:
  repeat: 2         # repeats whole experiment list. Order: 1, 2, 1, 2
  queue: null       # if valid path, create an experiment queue available to modify on the hard drive
  
                    # non-global parameters can be set separately for each experiment
Repeat: 1           # copies a single experiment many times **before** fancy parsing. Resulting order: 1, 1, 2, 2
Name: test          # if Name==skip, experiment will be skipped 
YamlLog: log.yaml   # experiment definition with additional information (added by Module) will be saved there
Module: modules.example      # location of the module that will be run by run.py
```

There are conventions for parameter names (this is not required):

* `parameter` is used (required or not) in modules
* `Parameter` is used in `run.py` for experiment parsing and saving logs
* `PARAMETER` is added by scripts, not by user, and should be used to write results, e.g. `ACCURACY`
* `_parameter` to denote something that is not used by the code, but might be used in fancy parsing

Custom modules will require custom parameters. It is advised to write flexible module with many parameters.

# Running experiments

Script `run.py` launches experiments specified in experiment definition.

* If `Global.queue` parameter is specified as a valid path, the Queue of the experiments will be stored as a `.yaml` file and can be modified when experiments are running. Otherwise, Queue is stored in RAM memory and cannot be modified.


* If experiment is stopped with `KeyboardInterrupt`, there will be 2 second pause during which `run.py` can be interrupted completely by a second `KeyboardInterrupt`. If not interrupted completely, next experiment in the Queue will start.

* If `Name: skip`, training will not be performed, but experiment parameters can be used in fancy parsing. Skipped experiments will not launch the Module.

* `run.py` has its own command line arguments. Modules might contain their own command line arguments and to use them, you should pass them to `run.py` and it will pass them further down to the Module. If argument name is used by both `run.py` and the Module, it will be used by both.
   ```
   > python run.py --help
   optional arguments:
     -h, --help            show this help message and exit
     --dry                 Skip execution but parse experiments
     --exp EXP             Path to .yaml file with experiments
     --pick PICK, --cherrypick-experiments PICK
                           Run only selected experiments, e.g. 0,1,3 or 1
   ```

* You can update default config straight from command line by passing arguments with `+` prefix instead of `-`, e.g. `python run.py +Global.queue=queue.yaml` without any spaces. Use quotations if needed.

# Modules

Module is a package that provides a function accepting one argument `exp` which is a dict-like structure with details defined in experiment definition (e.g. `experiment.yaml`). They contain code for running an actual experiment. They use experiments parsed by `run.py`. Besides these parameters, they might define their own command line arguments. It is recommended that only a few settings related to hardware are defined as command line arguments, and the rest are in experiment definition.

[Available modules](modules/README.md)

# python run.py

`run.py` is parsing the experiments and launching a Module to do its work. Path to the module should be specified as a parameter `Module` in default config of experiment definition. Before Module starts doing what it is supposed to do, experiments will be parsed and there are a few tricks to it.

## Fancy parsing with `parse`

> **WARNING**: Python's `eval` is considered unsafe - if someone has access to your experiment definition, they can run malicious code

All values for experiments can be specified explicitly in `experiment.yaml` as a number, string, dictionary or list, e.g. `sparsity: 0.9`, but there are some tricks that can simplify longer and complicated experiments. Fancy parsing allows you to execute python code during parsing. To do this, you need to type `parse` in the beginning of a parameter value. With this, you can do really cool tricks.

#### Tricks:

1. `odd_name: parse '\'.join([directory, Name])` will run real Python code and the result will be saved as `odd_name`. Code will be executed using `parse` function with variable scope from current experiment. In special case, if `directory` is fancy-parsed too, `directory` should be resolved before `odd_name`. Default config values can be fancy-parsed too, but they are resolved at the end.

2. `sparsity: parse 0.5 * E[-1].sparsity` will be parsed as half of the `sparsity` value from the previous experiment in the experiment Queue. This works, because before running `parse`, list `E` is added to the scope and this allows access to previous experiments.

3. You can access values from nested dictionaries using `.` or standard indexing:

```
pruning_config:
   odd_config:
      sparsity: 0.5
---
sparsity: parse E[-1].pruning_config.odd_config.sparsity
pruning_config:
   sparsity: parse E[-1]['pruning_config']['odd_config']['sparsity']
```

4. You can access values from parents' levels:

```
abc: 1
nested:
   abc: 2
   nested2:
      test: parse abc
```

This will work and will resolve to `test: 2`. Deeper levels have priority in this case.

5. Other examples:

```
Name: test
---
model: VGG
Name: test2
directory: parse f"{Name}/{E[-1].Name}/{model}"
```

> Resulting `directory` value will be `test2/test/VGG`.

```
list: [1, 2, 3]
---
list: [2, 3, 4]
number2: parse list[0]
number3: parse E[-1].list[2]
```

> Resulting `number2` value will be `2` and `number3` value will be `3`.

```
nested:
   test: 1

incremented: parse nested.test + 1
```

> Resulting `incremented` value will be `2`.

```
# DEFAULT CONFIG
test1: parse test3 - 1
---
test2: 5
test3: parse test2 - 1
```

> This is correct. `parse` in default config is resolved at the end - `test3: 4`, `test1: 3`.

Following examples **won't work**

```
test3: parse test2 - 1  # ERROR: test2 IS UNKNOWN (WRONG ORDER)
test2: 5
```

```
# DEFAULT CONFIG
test1: 1
---
test2: parse test1 # ERROR: test1 IS UNKNOWN (CAN'T REFERENCE DEFAULT CONFIG)
```

Order is important for fancy parsing. If you need to reference a default value, redefine it in the experiment instead.

## Differences between `parse` and `solve`

## Logs management

Modules are encouraged to leave meaningful metrics in short format. To do so, they should add their results to the (dict-like) experiment, which will save all received data in `experiment.yaml/YamlLog`. These logs will contain both experiment formulation and the results. There are some tools that can make it easier to deal with large number of experiments...

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

Can be used to apply a query on `.yaml` log file. Available operations are listed in arguments. Arguments (query) can be read from a file using `%FILENAME`.

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

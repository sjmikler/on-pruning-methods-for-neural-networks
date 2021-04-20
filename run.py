import argparse
import importlib
import pprint
import time

from tools import parser, utils
from tools.utils import cprint

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--dry",
                        action='store_true',
                        help="Skip training but parse experiments")
arg_parser.add_argument("--no-memory-growth",
                        action='store_true',
                        help="Disables memory growth")
arg_parser.add_argument("--gpu",
                        default=None,
                        type=str,
                        help="Which GPUs to use during training, e.g. 0,1,3 or 1")
arg_parser.add_argument("--pick",
                        "--cherrypick-experiments",
                        default=None,
                        type=str,
                        help="Run only selected experiments, e.g. 0,1,3 or 1")
args, unknown_args = arg_parser.parse_known_args()
import tensorflow as tf

if args.gpu is not None:
    gpus = tf.config.get_visible_devices('GPU')
    gpu_indices = [num for num in range(10) if str(num) in args.gpu]
    gpus = [gpus[idx] for idx in gpu_indices]
    tf.config.set_visible_devices(gpus, 'GPU')


if not args.no_memory_growth:
    utils.set_memory_growth()

default_config, experiment_queue = parser.load_from_yaml(yaml_path="experiment.yaml",
                                                         unknown_args=unknown_args)
utils.set_precision(default_config.precision)

for exp_idx, exp in enumerate(experiment_queue):
    if args.pick is not None:
        if str(exp_idx) not in args.pick:
            cprint(f"SKIPPING EXPERIMENT {exp_idx}")
            continue

    cprint("EXPERIMENT:")
    pprint.pprint(exp)
    print()
    if args.dry:
        continue
    if exp.name == "skip":
        cprint("SKIPPING TRAINING")

    module = importlib.import_module(exp.module)
    try:
        module.main(exp)
    except KeyboardInterrupt:
        cprint("\n\nSKIPPING EXPERIMENT, WAITING 2 SECONDS BEFORE RESUMING...")
        time.sleep(2)

if isinstance(experiment_queue, parser.YamlExperimentQueue):
    cprint(f"REMOVING QUEUE {experiment_queue.path}")
    experiment_queue.close()

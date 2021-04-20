import argparse
import importlib
import pprint
import time

from tools import parser, utils

cprint = utils.get_cprint(color='red')

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--exp",
                        default='experiment.yaml',
                        type=str,
                        help="Path to .yaml file with experiments")
arg_parser.add_argument("--dry",
                        action="store_true",
                        help="Skip execution but parse experiments")
arg_parser.add_argument("--pick",
                        "--cherrypick-experiments",
                        default=None,
                        type=str,
                        help="Run only selected experiments, e.g. 0,1,3 or 1")
args, unknown_args = arg_parser.parse_known_args()
if unknown_args:
    cprint(f"UNKNOWN CMD ARGUMENTS: {unknown_args}")

default_config, experiment_queue = parser.load_from_yaml(yaml_path=args.exp,
                                                         unknown_args=unknown_args)

for exp_idx, exp in enumerate(experiment_queue):
    if args.pick is not None:
        if str(exp_idx) not in args.pick:
            cprint(f"SKIPPING EXPERIMENT {exp_idx}")
            continue

    print()
    cprint("NEW EXPERIMENT:")
    pprint.pprint(exp)
    if args.dry:
        continue
    if exp.name == "skip":
        cprint("SKIPPING TRAINING")

    module = importlib.import_module(exp.module)
    exp.reset_unused_parameters(exclude=['GLOBAL_REPEAT', 'REP', 'REPEAT',
                                         'RND_IDX', 'queue', 'module'])
    try:
        module.main(exp)
    except KeyboardInterrupt:
        cprint("\n\nSKIPPING EXPERIMENT, WAITING 2 SECONDS BEFORE RESUMING...")
        time.sleep(2)

if isinstance(experiment_queue, parser.YamlExperimentQueue):
    cprint(f"REMOVING QUEUE {experiment_queue.path}")
    experiment_queue.close()

import argparse
import os
import time

import yaml

import tools

print = tools.get_cprint(color='red')

arg_parser = argparse.ArgumentParser(prefix_chars='-+')
arg_parser.add_argument("--dry",
                        action="store_true",
                        help="skip execution but parse experiments")
arg_parser.add_argument("--exp",
                        default='experiment.yaml',
                        type=str,
                        help="path to .yaml file with experiments")
arg_parser.add_argument("--pick",
                        type=int,
                        nargs='*',
                        help="run only selected experiments, e.g. 0 1 3 or just 1")
args, unknown_args = arg_parser.parse_known_args()
print(f"UNKNOWN CMD ARGUMENTS: {unknown_args}")
print(f"  KNOWN CMD ARGUMENTS: {args.__dict__}")

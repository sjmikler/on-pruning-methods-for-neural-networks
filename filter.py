import argparse
import datetime
import os

import yaml

import tools.constants as C
from tools.utils import get_cprint

cprint = get_cprint('green')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filtering and sorting of .yaml logs.",
                                     fromfile_prefix_chars='%')
    parser.add_argument('path', type=str, nargs='*',
                        help='Path to .yaml file containing logs')
    parser.add_argument('-d', '--dest', type=str, default='yaml_logs',
                        help='Directory of new .yaml file')
    parser.add_argument('-f', '--filter', type=str, action='append',
                        help='Python lambda function accepting experiment dict and '
                             'returning boolean')
    parser.add_argument('-s', '--sort', type=str, action='append',
                        help='Python lambda function accepting experiment dict and '
                             'returning sorting values')
    args = parser.parse_args()

    logs = []
    for path in args.path:
        with open(path, 'r') as f:
            log = yaml.safe_load_all(f)
            logs.extend(log)
    cprint(f"Loaded {len(logs):^5} logs!")

    if args.filter:
        for f in args.filter:
            logs = filter(eval(f), logs)
            logs = list(logs)
    cprint(f"Filter {len(logs):^5} logs!")

    if args.sort:
        for f in args.sort:
            logs = sorted(logs, key=eval(f))

    os.makedirs(args.dest, exist_ok=True)
    now = datetime.datetime.now().strftime(C.time_formats[1])
    dest = os.path.join(args.dest, f'flogs_{now}.yaml')
    with open(dest, 'w') as f:
        yaml.safe_dump_all(logs, f)
    cprint(f"SAVED: {dest}")

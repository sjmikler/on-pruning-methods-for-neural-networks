import argparse
import datetime
import os

import yaml

from tools import constants as C, utils

print = utils.get_cprint('green')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filtering and sorting of .yaml "
                                                 "logs. Use %FILENAME to load "
                                                 "arguments from file.",
                                     fromfile_prefix_chars='%')
    parser.add_argument('path', type=str, nargs='*',
                        help='path to .yaml file containing logs')
    parser.add_argument('--dest', type=str, default='yaml_filter',
                        help='directory of new .yaml file')
    parser.add_argument('--filter', type=str, action='append',
                        help='python lambda function accepting experiment dict and '
                             'returning boolean')
    parser.add_argument('--sort', type=str, action='append',
                        help='python lambda function accepting experiment dict and '
                             'returning sorting values')
    parser.add_argument('--reverse', action='store_true',
                        help='reverse sorting order')
    parser.add_argument('--keep-keys',
                        type=str,
                        nargs='*',
                        help='keys in the experiment dict that should be kept, '
                             'skip to keep everything')
    args = parser.parse_args()

    logs = []
    for path in args.path:
        with open(path, 'r') as f:
            log = yaml.safe_load_all(f)
            logs.extend(log)
    print(f"Loaded {len(logs):^5} logs!")

    if args.filter:
        for f in args.filter:
            logs = filter(eval(f), logs)
            logs = list(logs)
        print(f"Filter {len(logs):^5} logs!")

    if args.sort:
        for f in args.sort:
            logs = sorted(logs, key=eval(f), reverse=args.reverse)

    if args.keep_keys:
        nlogs = []
        for log in logs:
            log = {k: log[k] for k in args.keep_keys if k in log}
            nlogs.append(log)
        logs = nlogs

    os.makedirs(args.dest, exist_ok=True)
    now = datetime.datetime.now().strftime(C.time_formats[1])
    dest = os.path.join(args.dest, f'flogs_{now}.yaml')
    with open(dest, 'w') as f:
        yaml.safe_dump_all(logs, f, explicit_start=True, sort_keys=False)
    print(f"SAVED: {dest}")

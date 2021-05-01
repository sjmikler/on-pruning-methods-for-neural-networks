import argparse
import datetime
import os

import yaml
import yaml.reader

from tools import constants as C, utils

print = utils.get_cprint(color='green')


def recursive_collect_logs(path, exclude, verbose=False, level=0):
    logs = []
    for x in os.listdir(path):
        if x in exclude:
            continue

        if x.endswith('.yaml'):
            full_path = os.path.join(path, x)
            try:
                with open(full_path, 'r') as f:
                    for exp in yaml.safe_load_all(f):
                        if exp and exp not in logs:
                            logs.append(exp)
            except Exception as e:
                print(f"EXCEPTION WHEN READING {full_path}")
                print(e)
                continue
        if os.path.isdir(npath := os.path.join(path, x)):
            for exp in recursive_collect_logs(npath,
                                              exclude=exclude,
                                              verbose=verbose,
                                              level=level + 1):
                if exp not in logs:
                    logs.append(exp)
    if len(logs) and (verbose or level == 0):
        print(f"{'  ' * level}Found {len(logs):^5} logs under {path}")
    return logs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Recursive gathering of .yaml logs.")
    parser.add_argument('path', type=str, default=['.'], nargs='*',
                        help='starting directories for recursive log collecting')
    parser.add_argument('--exclude', type=str, default=[], nargs='*',
                        help='skip directories or files')
    parser.add_argument('--dest', type=str, default='yaml_collect',
                        help='directory of new .yaml file')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='print visited directories during recursive collecting')
    args = parser.parse_args()

    logs = []
    for path in args.path:
        log = recursive_collect_logs(path, exclude=args.exclude, verbose=args.verbose)
        logs.extend(log)

    os.makedirs(args.dest, exist_ok=True)
    now = datetime.datetime.now().strftime(C.time_formats[1])
    dest = os.path.join(args.dest, f'logs_{now}.yaml')
    with open(dest, 'w') as f:
        yaml.safe_dump_all(logs, f, explicit_start=True, sort_keys=False)
    print(f"SAVED: {dest}")

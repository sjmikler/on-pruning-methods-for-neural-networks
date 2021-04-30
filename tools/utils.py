import datetime
import pprint
import socket

from tools import constants as C


def get_cprint(color):
    def cprint(*args, **kwargs):
        if not args:
            print(**kwargs)
            return

        args = list(args)
        args[0] = C.color2code[color] + '# ' + str(args[0])
        args[-1] = str(args[-1]) + C.color2code['reset']
        print(*args, **kwargs)

    return cprint


class Experiment:
    """Dict like structure that allows for dot indexing."""
    _internal_names = ['dict', '_usage_counts', '_ignored_counts']

    def __init__(self, from_dict={}):
        self._usage_counts = {key: 0 for key in from_dict}
        self._ignored_counts = set()

        self.dict = {k: Experiment(v) if isinstance(v, dict) else v for k, v in
                     from_dict.items()}

    def __setattr__(self, key, value):
        if key in self._internal_names:
            super().__setattr__(key, value)
        else:
            self.dict[key] = value
            self._usage_counts[key] = 0

    def __getattr__(self, key):
        # complicated for `deepcopy` to work
        # fallbacks to normal dictionary
        if key in super().__getattribute__('dict'):
            self._usage_counts[key] += 1
            return self.dict[key]
        else:
            return self.dict.__getattribute__(key)

    def __getitem__(self, item):
        if item in self.dict:
            self._usage_counts[item] += 1
            return self.dict[item]
        else:
            raise KeyError(item)

    def __setitem__(self, key, value):
        self.dict[key] = value
        self._usage_counts[key] = 0

    def __iter__(self):
        for key in self.dict.keys():
            yield key

    def __str__(self):
        return pprint.pformat(self.todict(), sort_dicts=False)

    def __contains__(self, item):
        return self.dict.__contains__(item)

    def _reset_usage_counts(self, ignore_keys):
        self._usage_counts = {key: 0 for key in self.dict}
        self._ignored_counts = set(ignore_keys)

    def update(self, other):
        for key in other.keys():
            self._usage_counts[key] = 0
        self.dict.update(other)

    def get_unused_parameters(self):
        unused_keys = []
        for key in self.dict.keys():
            if key in self._ignored_counts:
                continue

            counts = self._usage_counts[key]
            if counts == 0:
                unused_keys.append(key)
        return unused_keys

    def todict(self):
        new_dict = {}
        for key, value in self.dict.items():
            if isinstance(value, Experiment):
                value = value.todict()
            if hasattr(value, 'tolist'):  # for numpy objects
                value = value.tolist()
            new_dict[key] = value
        return new_dict


def filter_argv(argv: list, include: list, exclude: list):
    filtered = []
    adding = False
    for arg in argv:
        if arg[0] in include:
            adding = True
        if arg[0] in exclude:
            adding = False

        if adding:
            filtered.append(arg)
    return filtered


def parse_time(strtime):
    for format in C.time_formats:
        try:
            return datetime.datetime.strptime(strtime, format)
        except ValueError:
            continue
    raise Exception("UNKNOWN TIME FORMAT!")


class SlackLogger:
    def __init__(self, config):
        self.config = config
        self.token = config.token
        self.messages = []

    def add_exp_report(self, exp):
        message = eval(self.config.say, {}, {'exp': exp})
        message = "`" + message + "`"
        self.messages.append(message)

    def add_finish_report(self):
        host = socket.gethostname()
        message = f"Experiment on {host} is completed!"
        self.messages.append(message)

    def send_all(self):
        import slack
        client = slack.WebClient(self.token)
        final_message = '\n'.join(self.messages)

        failed = False
        try:
            client.chat_postMessage(channel=self.config.channel,
                                    text=final_message)
        except slack.errors.SlackApiError as e:
            failed = True
            print(e)
        return failed

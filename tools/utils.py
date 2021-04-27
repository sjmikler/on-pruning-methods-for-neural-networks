import datetime
from copy import deepcopy

import tools.constants as C


def get_cprint(color):
    def cprint(*args, **kwargs):
        print('#',
              C.color2code[color],
              *args,
              C.color2code['reset'],
              **kwargs)

    return cprint


class ddict(dict):
    def __init__(self, *args, **kwargs):
        """Dict with dot syntax. Keeps track of unused keys."""
        super().__init__(*args, **kwargs)
        self.__dict__['_unused_parameters'] = set()
        self.reset_unused_parameters()  # set everything as unused

        for key, value in self.items():
            if isinstance(value, dict):
                self.__setattr__(key, ddict(value))

    def __getattr__(self, key):
        if key in self:
            if key in self.__dict__['_unused_parameters']:
                self.__dict__['_unused_parameters'].remove(key)
            return self[key]
        else:
            return getattr(super(), key)

    def __setattr__(self, key, val):
        if isinstance(val, dict):
            val = ddict(val)
        self[key] = val
        self.__dict__['_unused_parameters'].add(key)

    def reset_unused_parameters(self, exclude=()):
        for k, v in self.items():
            if k in exclude:
                if k in self.__dict__['_unused_parameters']:
                    self.__dict__['_unused_parameters'].remove(k)
                continue
            self.__dict__['_unused_parameters'].add(k)

    def get_unused_parameters(self):
        return tuple(self.__dict__['_unused_parameters'])


def unddict(d):
    """Recursivly transform ddicts into dicts"""
    d = deepcopy(d)
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = unddict(v)
    return dict(d)


def contains_any(t, *opts):
    return any([x in t for x in opts])


def parse_time(strtime):
    for format in C.time_formats:
        try:
            return datetime.datetime.strptime(strtime, format)
        except ValueError:
            continue
    raise Exception("UNKNOWN TIME FORMAT!")


def get_date_from_exp(exp):
    if 'time' in exp:
        return parse_time(exp['time'])
    if 'TIME' in exp:
        return parse_time(exp['TIME'])
    else:
        return datetime.datetime.min


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

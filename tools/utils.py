from copy import deepcopy


def get_cprint(color):
    color2code = {
        'black': '\x1b[30m',
        'red': '\x1b[31m',
        'green': '\x1b[32m',
        'yellow': '\x1b[33m',
        'blue': '\u001b[34m',
        'magenta': '\u001b[35m',
        'cyan': '\u001b[36m',
        'white': '\x1b[37m',
        'light red': '\u001b[31;1m',
        'light green': '\u001b[32;1m',
        'light yellow': '\u001b[33;1m',
        'light blue': '\u001b[34;1m',
        'light magenta': '\u001b[35;1m',
        'light cyan': '\u001b[36;1m',
        'light white': '\u001b[37;1m',
        'reset': '\x1b[0m',
    }

    def cprint(*args, **kwargs):
        print('#', color2code[color], *args, color2code['reset'], **kwargs)

    return cprint


class ddict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__['_unused_parameters'] = set()
        self.reset_unused_parameters()

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

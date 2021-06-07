import datetime
from tools import constants as C


def get_cprint(color):
    def cprint(*args, **kwds):
        if not args:
            print(**kwds)
            return

        args = list(args)
        args[0] = C.color2code[color] + '# ' + str(args[0])
        args[-1] = str(args[-1]) + C.color2code['reset']
        print(*args, **kwds)

    return cprint


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
    for time_format in C.time_formats:
        try:
            return datetime.datetime.strptime(strtime, time_format)
        except ValueError:
            continue
    raise Exception("UNKNOWN TIME FORMAT!")

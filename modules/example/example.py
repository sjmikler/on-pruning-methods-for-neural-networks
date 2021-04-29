import random
import time

from ._initialize import *


def main(exp):
    print(f"I am an example of an experiment!")
    print(f"I will show you random keys from your experiment definition!")

    keys = list(exp.keys())
    for i in range(3):
        time.sleep(1)
        key = random.choice(keys)
        print(f"{i+1}/3:\n{key}: {exp[key]}")

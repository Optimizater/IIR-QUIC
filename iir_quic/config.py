"""
This module provides tools for printing debugging information and some parameter settings for the project.
"""

import datetime

DEBUG = False     # A flag to enable or disable debug mode.
MAKE_LOG = False  # A flag to enable or disable log.

NONZERO = 1e-8
SCAD_A = 3.7
MCP_GAMMA = 2
EBIC_GAMMA = 0.5

RAND_SEED = 1


def printDecorate(func, color="black"):
    """A decorator function that adds color to the output of the print function."""
    assert callable(func) and func.__name__ == "print", "The function must be print()"

    _color = {
        "black": 30,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "purple": 35,
        "cyan": 36,
        "white": 37,
    }

    def wrapper(*args, **kwargs):
        if DEBUG:
            args = (f"\033[{_color[color]}m[{datetime.datetime.now().strftime('%H:%M')}] DEBUG:\033[0m ",) + args
        result = func(*args, **kwargs)
        return result

    return wrapper


def debugPrint(*args, **kwargs):
    """Prints debug messages in blue color."""
    if DEBUG: printDecorate(print, "blue")(*args, **kwargs)


def importantPrint(*args, **kwargs):
    """Prints important messages in red color."""
    if DEBUG: printDecorate(print, "red")(*args, **kwargs)


def warningPrint(*args, **kwargs):
    """Prints warning messages in yellow color."""
    if DEBUG: printDecorate(print, "yellow")(*args, **kwargs)

def attentionPrint(*args, **kwargs):
    """Prints warning messages in purple color."""
    if DEBUG: printDecorate(print, "purple")(*args, **kwargs)


# Print the current time when the module is imported, if not already printed
# global _time_printed
# _time_printed = False
# if not _time_printed:
#     print(f"\033[34m{'-' * 50}")
#     print(f"{'Debug' if DEBUG else 'Release'} time: {datetime.datetime.now()}")
#     print(f"{'-' * 50}\033[0m")
#     _time_printed = True

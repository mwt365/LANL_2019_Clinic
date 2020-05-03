
from importlib import __import__ as imp
import inspect
import os
import re
import sys

from pnspipe import PNSPipe

# This dictionary is used to register all available pipeline functions
_pipe_functions = {
    x: [] for x in "preprocess;onsegment;postsegment;postprocess".split(';')}


def register_pipe_function(f):
    """
    Call register_pipe_function on a function that is designed to accept
    a PNSPipe object and key word arguments. Some examples are present
    in this file, but the register_pipe_function allows functions in
    other files to register their processing functions so that they can
    be included in the list of available such functions supplied by
    the help commanmd and so that they can be called from the main loop.

    The functions defined in the present file are included automatically
    and need not be registered manually.
    """
    if not inspect.isfunction(f):
        return

    params = inspect.signature(f).parameters
    keys = list(params.keys())
    annotes = [x.annotation for x in params.values()]

    try:
        if len(keys) == 2:
            # the first argument should be a PNSPipe;
            # the second should be named kwargs
            if annotes[0] == PNSPipe and keys[1] == 'kwargs':
                _pipe_functions['onsegment'].append(f)

            if annotes[0] == list and keys[1] == 'kwargs':
                _pipe_functions['postsegment'].append(f)

        if len(keys) == 3:
            if annotes[0] == re.Pattern \
               and annotes[1] == re.Pattern \
               and keys[2] == 'kwargs':
                _pipe_functions['preprocess'].append(f)

        if len(keys) == 4:
            if annotes[0] == str \
                and annotes[1] == dict \
                    and annotes[2] == dict:
                _pipe_functions['postprocess'].append(f)

    except Exception as eeps:
        print(eeps)


def describe_pipe_functions(key: str):
    """
    Prepare a long text string describing all the registered
    pipe functions satisying the given key. If key is "",
    describe them all.
    """
    fstring = "{0}: {1}"
    if key == "":
        lines = []
        for key, vals in _pipe_functions.items():
            lines.append(f"{key.upper()} ------------------\n")
            lines.extend([fstring.format(x.__name__, x.__doc__)
                          for x in vals])
        lines.append("\n")
    else:
        lines = [fstring.format(x.__name__, x.__doc__) for x in
                 _pipe_functions.vals()]

    return "\n".join(lines)


folder = os.path.split(__file__)[0]
for file in os.listdir(folder):
    path = f"Pipeline.{os.path.splitext(file)[0]}"
    try:
        imp(path)
        for name, obj in inspect.getmembers(sys.modules[path]):
            register_pipe_function(obj)
    except:
        pass


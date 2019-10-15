#plotter_test.py

import pytest
from plotter import *
from os import path


assert path.exists('xml color map files'), 'file not created'
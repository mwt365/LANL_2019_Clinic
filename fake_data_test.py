#fake_data_test.py
#tests fake_data.py

import pytest
from fake_data import *
import os.path
from os import path

make_dig_file('fake_data_test.dig',
        [0, 1e-5, 2e-5, 4e-5],
        [1000, 1000, 5000, 3000]
    )

assert path.exists('fake_data_test.dig'), 'file not created'

os.remove('fake_data_test.dig')
#spectrogram_test.py

from spectrogram import *
from fake_data import *

make_dig_file('fake_data_test.dig',
        [0, 1e-5, 2e-5, 4e-5],
        [1000, 1000, 5000, 3000]
    )

test = Spectrogram("fake_data_test.dig")

assert 3.98e-5 < test.t_final < 4.001e-5, "t_final failed"

assert test.v_max ==  19375, "v_max failed"

assert test.digfile, "digfile failed, OR is not dig file"

assert test.point_number(1e-5) == 5e5, "point_number failed"

# assert test.values(test.t0,4)[3] == 0.27630616932, "values failed"



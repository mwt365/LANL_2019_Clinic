#! /usr/bin/env python3

import os
import sys
import re
import numpy as np


class DigFile:
    """
    Representation of a photon Doppler velocimetry file stored in 
    the .dig format. On creation, the file header is read and processed;
    information in the top 512 bytes is stored in a notes dictionary.
    Information from the second 512-byte segment is decoded to infer
    the number of data points, the number of bytes per point, the 
    start time and sampling interval, and the voltage scaling.

    The actual data remain on disk and are loaded only as required either
    to generate a spectrogram for a range in time or a spectrum from a 
    shorter segment. The values are loaded from disk and decoded using
    the *values* method which takes a start time and either an end time
    or an integer number of points to include. Alternatively, the raw values
    may be returned with the *raw_values* method. For either, the corresponding
    sample times are available from *time_values*.
    """

    def __init__(self, filename):
        """
        We assume that the first 1024 bytes of the file contain ascii text
        describing the data in the file. The first 512 bytes may vary, but
        the second 512-byte chunk should include (in order) the following:
            the number of samples
            the format (8, 16, or 32)
            the sample period (s)
            the start time
            the voltage step
            offset voltage
        """
        self.filename = filename
        self.path = os.path.realpath(filename)
        _, ext = os.path.splitext(filename)
        self.ext = ext.lower()[1:]
        # Let's declare all the fields we expect the Spectrogram to hold
        self.t0 = 0.0
        self.dt = 0.0
        self.V0 = 0.0
        self.dV = 0.0
        self.bits = 8
        self.num_samples = 0
        self.notes = dict()
        if self.ext == 'dig':
            self.load_dig_file()
        else:
            raise ValueError(f"I don't understand files with extension {ext}")

    @property
    def t_final(self):
        return self.t0 + self.dt * (self.num_samples - 1)

    @property
    def frequency_max(self):
        return 0.5 / self.dt

    def load_dig_file(self):
        """
        A .dig file has a 1024-byte ascii header, typically followed by binary
        data. The first 512-byte segment has information specific to the
        particular device recording the data. The second 512-byte segment
        holds a list of values separated by carriage return-linefeed
        delimiters. The remaining space is padded with null bytes.
        """
        headerLength = 1024
        text = open(self.path, 'rb').read(headerLength).decode('ascii')
        top = [x.strip() for x in text[:headerLength // 2].replace(
            "\000", "").split('\r\n') if x.strip()]
        self.decode_dig_header(top)

        bottom = [x.strip() for x in text[headerLength // 2:].replace(
            "\000", "").split('\r\n') if x.strip()]
        self.headers = top

        # These fields are pretty short, but I think they're quite
        # descriptive. Do people think we should bulk them up?

        self.V0 = float(bottom[-1])  # voltage offset
        self.dV = float(bottom[-2])  # voltage interval
        self.t0 = float(bottom[-3])  # initial time
        self.dt = float(bottom[-4])  # sampling interval
        self.bits = int(bottom[-5])  # 8, 16, or 32
        self.num_samples = int(bottom[-6])
        self.bytes_per_point = self.bits // 8
        self.data_format = np.dtype({1: np.uint8, 2: np.int16,
                                     4: np.int32}[self.bytes_per_point])
        if self.bits > 8:
            self.set_data_format()

    def set_data_format(self):
        """
        Determine the endianness of the data and adjust the data_format accordingly.
        """
        if 'byte_order' in self.notes:
            native_order = 'MSB' if sys.byteorder == 'little' else 'LSB'
            if native_order != self.notes['byte_order']:
                self.swap_byte_order()
        else:
            # We need to try both and figure it out! The current strategy is
            # to look at the absolute differences between successive elements
            # for both big-endian and little-endian interpretation. Whichever
            # produces the smaller average jump between values is assumed
            # to be the right ordering.
            order = [0, 1]
            nvals = 10000
            for n in range(2):
                vals = self.raw_values(None, nvals)
                shifts = np.abs(vals - np.roll(vals, 1))
                # we should omit the two end points in computing the mean
                order[n] = np.mean(shifts[1:-2])
                self.swap_byte_order()
            if order[0] > order[1]:
                self.swap_byte_order()  # we want to go with the smaller average

    def swap_byte_order(self):
        self.data_format = self.data_format.newbyteorder('S')

    def decode_dig_header(self, tList):
        """
        Maybe some help here?
        """
        instrument_spec_codes = {
            'BYT_N': 'binary_data_field_width',
            'BIT_N': 'bits',
            'ENC': 'encoding',
            'BN_F': 'number_format',
            'BYT_O': 'byte_order',
            'WFI': 'source_trace',
            'NR_P': 'number_pixel_bins',
            'PT_F': 'point_format',
            'XUN': 'x_unit',
            'XIN': 'x_interval',
            'XZE': 'post_trigger_seconds',
            'PT_O': 'pulse_train_output',
            'YUN': 'y_unit',
            'YMU': 'y_scale_factor',
            'YOF': 'y_offset',
            'YZE': 'y_component',
            'NR_FR': 'NR_FR'
        }
        for line in tList:
            if any(s in line for s in instrument_spec_codes):
                for xstr in line.split(';'):
                    m = re.match(r'([^ ]*) (.*)', xstr)
                    if m and m.group(1) in instrument_spec_codes:
                        key, val = instrument_spec_codes[m.group(
                            1)], m.group(2)
                        # Attempt to decode the value
                        try:
                            val = int(val)  # can we convert to an integer?
                        except:
                            try:
                                # if not, can we convert to a float?
                                val = float(val)
                            except:
                                pass
                        # add the property to ourself
                        self.notes[key] = val
                        # print(f"{key} ==> {val}")

    def point_number(self, time):
        "Return the point number corresponding to the given time"
        return int((time - self.t0) / self.dt)

    def __str__(self):
        return "\n".join([
            self.filename,
            f"{self.bits} bits" + f" {self.notes['byte_order']} first" if 'byte_order' in self.notes else "",
            f"{self.t0*1e6} µs to {(self.t0 + self.dt*self.num_samples)*1e6} µs in steps of {self.dt*1e12} ps",
            f"{self.num_samples:,} points"
        ])

    def raw_values(self, t_start=None, ending=None):
        """
        Return a numpy array of the raw values present in this segment
        The ending argument can be an integer representing the number of points to include,
        or it can be a floating-point number indicating the ending time.
        """
        p_start, p_end = self._points(t_start, ending)
        n_samples = p_end - p_start + 1

        offset = 1024 + self.bytes_per_point * p_start
        with open(self.path, 'rb') as f:
            f.seek(offset, os.SEEK_SET)
            raw = np.fromfile(f, self.data_format, count=n_samples)
        return raw

    def values(self, t_start=None, ending=None):
        """
        Return a numpy array with the properly normalized voltages corresponding to this segment.
        """
        return self.V0 + self.dV * self.raw_values(t_start, ending)

    def time_values(self, t_start=None, ending=None):
        """
        Return an array of time values corresponding to this interval. The arguments
        are the same as for the values method.
        """
        p_start, p_end = self._points(t_start, ending)
        n_samples = p_end - p_start + 1
        if t_start == None:
            t_start = self.t0
        t_final = t_start + n_samples * self.dt
        return np.linspace(t_start, t_final, n_samples)

    def _points(self, t_start, ending):
        """
        Given a start time and either a number of points or an ending time,
        return a pair of integers specifying the point numbers of the first
        and last datum.
        """
        if t_start == None:
            pStart = 0
        else:
            pStart = self.point_number(t_start)
        if isinstance(ending, int):
            pEnd = pStart + ending - 1
        elif isinstance(ending, float):
            pEnd = self.point_number(ending)
        else:
            pEnd = self.num_samples - 1
        return (pStart, pEnd)

    def thumbnail(self, t_start=None, t_end=None, points=1000, stdev=False):
        """
        Produce a condensed representation of the data by processing chunks
        to yield mean (and standard deviation) as a function of time.
        """
        p_start, p_end = self._points(t_start, t_end)
        if t_start == None:
            t_start = p_start * self.dt + self.t0
        t_end = p_end * self.dt + self.t0
        chunk_size = (p_end - p_start + 1) // points
        numpts = points * chunk_size
        chunks = self.raw_values(t_start, numpts)
        chunks = chunks.reshape((points, chunk_size))
        means = chunks.mean(axis=1)
        times = np.linspace(t_start, t_start + self.dt *
                            numpts, points)
        d = dict(
            times=times,
            means=means,
            mins=chunks.min(axis=1),
            maxs=chunks.max(axis=1)
        )
        if stdev:
            d['stdevs'] = chunks.std(axis=1)
        # Now create a quick-and-dirty thumbnail that
        # takes successive points alternatively from mins
        # and maxs, that can be plotted against times to roughly
        # describe the envelope of values
        d['peak_to_peak'] = np.asarray([
            d['mins'][n] if n % 2 else d['maxs'][n] for n in range(len(times))
        ])
        return d


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    for file in os.listdir('../dig/'):
        filename = os.path.splitext(file)[0]
        if filename != 'GEN3_CHANNEL1KEY001':
            continue
        df = DigFile(f'../dig/{filename}.dig')
        print(df)
        thumb = df.thumbnail(0, 1e-3, stdev=True)
        xvals = thumb['times']
        yvals = thumb['stdevs'] # thumb['peak_to_peak']
        plt.plot(xvals * 1e6, yvals)
        plt.xlabel('$t (\\mu \\mathrm{s})$')
        plt.ylabel('amplitude')
        plt.title(filename.replace("_", ""))
        plt.savefig(f'../figs/thumbnails/{filename}.png')
        plt.show()

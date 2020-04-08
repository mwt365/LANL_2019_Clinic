# coding:utf-8

"""
::

  Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
  Purpose: Load a .dig file
  Created: 9/18/19
"""
import os
import sys
import re
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


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
    the **values** method which takes a start time and either an end time
    or an integer number of points to include. Alternatively, the raw values
    may be returned with the **raw_values** method. For either, the
    corresponding sample times are available from **time_values**.

    We assume that the first 1024 bytes of the file contain ascii text
    describing the data in the file. The first 512 bytes may vary, but
    the second 512-byte chunk should include (in order) the following:
    - the number of samples
    - the format (8, 16, or 32)
    - the sample period (s)
    - the start time
    - the voltage step
    - offset voltage

    It strongly seems that despite the documentation we received, the
    integers stored in the binary portion of the file are unsigned
    whether 8 or 16 bits. We have no 32-bit examples.
    """

    def __init__(self, filename):
        """
        """
        if not filename.endswith('.dig'):
            filename += '.dig'
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
        "The time of the last sample in the DigFile."
        return self.t0 + self.dt * (self.num_samples - 1)

    @property
    def frequency_max(self):
        r"""The Nyquist frequency, corresponding to :math:`\frac{1}{2 \Delta t}`,
        where :math:`\Delta t` is the sampling period (typically between 20 and
        50 ps).
        """

    @property
    def date(self):
        import re
        dateline = re.search(
            r"^(.*)(201\d)$", self.header_text, re.MULTILINE)
        if dateline:
            dt = dateline.group(0)
            if '=' in dt:
                return dt.split('=')[1].strip()
            return dateline.group(0)
        return ""

    @staticmethod
    def dig_dir():
        """
        Return the path to the directory of dig files
        """
        root = "LANL_2019_Clinic"  # This is the name of the root folder for the source.
        # We are operating under the assumption that the data files (dig files) will be
        # in a folder that is on the same level as this source folder.
        parent, curr = __file__, ""
        while curr != root:
            parent, curr = os.path.split(parent)
        diggers = os.path.join(parent, 'dig')
        return os.path.realpath(diggers)

    @property
    def basename(self):
        "Return the name of this file, without extension"
        return os.path.splitext(os.path.split(self.path)[1])[0]

    @property
    def title(self):
        "Name of the file without extension, but including folder for segments"
        head, file = os.path.split(self.path)
        file_or_seg = os.path.splitext(file)[0]  # discard the extension
        if self.is_segment:
            file_or_seg = os.path.join(os.path.split(head)[1], file_or_seg)
        return file_or_seg.replace("_", "-")

    @property
    def rel_path(self):
        """
        Returns the relative path from the dig folder to the file
        """
        return os.path.relpath(self.path, start=self.dig_dir())

    @property
    def rel_dir(self):
        "Return the relative path to the directory holding this file"
        return os.path.split(self.rel_path)[0]

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

        htext = text.replace("\000", "").split("\r\n")
        adjusted = []
        for row in htext:
            if len(row) > 80:
                adjusted.append(row[:80])
                row = row[80:]
                while len(row) > 78:
                    adjusted.append("  " + row[:78])
                    row = row[78:]
                if row.strip():
                    adjusted.append("  " + row.strip())
            else:
                adjusted.append(row)
        self.header_text = "\n".join(adjusted)
        # These fields are pretty short, but I think they're quite
        # descriptive. Do people think we should bulk them up?

        self.V0 = float(bottom[-1])  # voltage offset
        self.dV = float(bottom[-2])  # voltage interval
        self.t0 = float(bottom[-3])  # initial time
        self.dt = float(bottom[-4])  # sampling interval
        self.bits = int(bottom[-5])  # 8, 16, or 32
        self.num_samples = int(bottom[-6])
        self.bytes_per_point = self.bits // 8
        # Should np.int32 be np.uint32? Don't know. No data.
        self.data_format = np.dtype(
            {1: np.uint8, 2: np.uint16, 4: np.uint32}[self.bytes_per_point])
        if self.bits > 8:
            self.set_data_format()

    def set_data_format(self):
        """
        Determine the endianness of the data and adjust the **data_format** accordingly.
        Most of the files we have seen have an entry of the form BYT_O: LSB in the
        first 512 bytes of the header. This seems to decode to the order of the bytes
        in the file being most-significant to least significant. If the file uses
        more than one byte per data point and this ordering disagrees with the native
        integer format, as determined by **sys.byteorder**, the data_format is swapped.
        In the case that no information can be found in the header, but more than one
        byte is used for storing the data, both orders are tried for the first
        nvals = 10000 values in the file and the ordering that yields the smoothest
        curve :math:`V(t)` is chosen.
        """
        if 'byte_order' in self.notes:
            order = "<" if self.notes['byte_order'] == 'LSB' else ">"
            self.data_format = np.dtype(f"{order}u{self.bits//8}")
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
                shifts = np.abs(np.asarray(
                    vals, dtype=np.float32) - np.roll(vals, 1))
                # we should omit the two end points in computing the mean
                order[n] = np.mean(shifts[1:-2])
                self.swap_byte_order()
            if order[0] > order[1]:
                self.swap_byte_order()  # we want to go with the smaller average

    def swap_byte_order(self):
        """Exchange the byte ordering in self.data_format."""
        self.data_format = self.data_format.newbyteorder('S')

    def decode_dig_header(self, tList):
        """
        TODO: I would like Trevor or Max (whoever wrote the original code) to
        supply some further information here.
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
        if len(tList) > 1:
            for line in tList:
                if any(s in line for s in instrument_spec_codes):
                    for xstr in line.split(';'):
                        m = re.match(r'([^ ]*) (.*)', xstr)
                        if m and m.group(1) in instrument_spec_codes:
                            key, val = instrument_spec_codes[m.group(
                                1)], m.group(2)
                            # Attempt to decode the value
                            try:
                                # can we convert to an integer?
                                val = int(val)
                            except:
                                try:
                                    # if not, can we convert to a float?
                                    val = float(val)
                                except:
                                    pass
                            # add the property to ourself
                            self.notes[key] = val
        else:
            try:
                self.header_args = [x.strip() for x in tList[0].split(',')]
            except:
                self.header_args = []

    def point_number(self, time):
        "Return the point number corresponding to the given time"
        return int((time - self.t0) / self.dt)

    def __str__(self):
        headerinfo = [f"{key}: {val}" for key, val in self.notes.items()]
        if hasattr(self, "header_args"):
            headerinfo.extend(self.header_args)
        datainfo = [
            self.filename,
            f"{self.bits} bits" + f" {self.notes['byte_order']} first" if 'byte_order' in self.notes else "",
            f"{self.t0*1e6} µs to {(self.t0 + self.dt*self.num_samples)*1e6} µs in steps of {self.dt*1e12} ps",
            f"{self.num_samples:,} points",
        ]
        datainfo.extend(headerinfo)
        return "\n".join(datainfo)

    def extract(self, filename, t_start, ending=None):
        """
        Save a .dig file with the portion of the data between the
        specified time limits.
        """
        p_start, p_end = self._points(t_start, ending)
        n_samples = p_end - p_start + 1
        with open(filename, 'w') as f:
            f.write(f"Extract from {self.filename}\r\n")
            f.write(f"{t_start} to {self.t0 + p_end * self.dt}\r\n")
            f.write(" " * (512 - f.tell()))
            # Now write the requisite dig file parameters
            f.write("\r\n".join(
                [str(x) for x in (n_samples, self.bits,
                                  self.dt, t_start,
                                  self.dV, self.V0)]))
#                                  (self.V0, self.dV, t_start,
#                                  self.dt, self.bits, n_samples)
#                                 ]))
            f.write(" " * (1024 - f.tell()))
        # Now open in binary mode and copy the data
        bytes_per_pt = self.bytes_per_point

        with open(filename, 'ab') as outfile:
            with open(self.path, 'rb') as infile:
                infile.seek(1024 + p_start * bytes_per_pt, os.SEEK_SET)
                outfile.write(infile.read(n_samples * bytes_per_pt))

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

    def microseconds(self, t_start=None, ending=None):
        """
        Return the time values for the range specified by t_start (in seconds!!)
        and ending (either as an int or a time (also in seconds)).
        """
        return self.time_values(t_start, ending) * 1e6

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
        chunks = self.values(t_start, numpts)
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

    @staticmethod
    def all_dig_files():
        """
        Walk the tree from ../dig down and harvest all .dig files.
        Return a list of filenames.
        """
        curdir = os.getcwd()
        os.chdir(DigFile.dig_dir())
        digfiles = []
        for base, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.dig'):
                    digfiles.append(os.path.join(base[2:], file))
        os.chdir(curdir)
        return sorted(digfiles)

    @staticmethod
    def inventory(segments=False):
        """Produce a pandas DataFrame showing all the .dig files in
        the /dig/ directory. If segments is True, show each segment file
        that was created by Fiducials.split(). By default, segments is False.
        """
        curdir = os.getcwd()
        home = DigFile.dig_dir()
        os.chdir(home)
        rows = []

        for filename in DigFile.all_dig_files():
            df = DigFile(filename)
            if not segments and df.is_segment:
                continue
            rows.append(dict(
                file=filename,
                date=df.date,
                bits=df.bits,
                dt=df.dt * 1e12,
                duration=df.dt * df.num_samples * 1e6
            ))
        os.chdir(curdir)
        return pd.DataFrame(rows)

    @property
    def is_segment(self):
        return self.header_text.startswith('Segment')


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    df = DigFile.inventory()
    print(df)

    df = DigFile('../dig/GEN3_CHANNEL1KEY001')
    print(df)

    tmp = df.fiducials()
    plt.plot(*tmp)
    plt.xlim(2.5e-6, 4e-6)
    plt.show()
    raise Exception("Done")

    for file in os.listdir('../dig/'):
        filename = os.path.splitext(file)[0]
        if filename != 'GEN3_CHANNEL1KEY001':
            continue
        df = DigFile(f'../dig/{filename}.dig')
        print(df)
        thumb = df.thumbnail(0, 1e-3, stdev=True)
        xvals = thumb['times']
        yvals = thumb['stdevs']  # thumb['peak_to_peak']
        plt.plot(xvals * 1e6, yvals)
        plt.xlabel('$t (\\mu \\mathrm{s})$')
        plt.ylabel('amplitude')
        plt.title(filename.replace("_", ""))
        plt.savefig(f'../figs/thumbnails/{filename}.png')
        plt.show()

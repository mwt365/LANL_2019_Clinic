#! /usr/bin/env python3

import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


class Spectrogram:
    """
    Representation of a photon Doppler velocimetry file stored in 
    the .dig format. On creation, the file header is read and processed;
    information in the top 512 bytes is stored in a notes dictionary.
    Information from the second 512-byte segment is decoded to infer
    the number of data points, the number of bytes per point, the 
    start time and sampling interval, and the voltage scaling.

    It is not completely transparent to me whether the voltage scaling
    information is useful to us. However, at the moment the integer samples
    are converted to floating point by scaling with the supplied voltage
    offset and step values.

    The actual data remain on disk and are loaded only as required either
    to generate a spectrogram for a range in time or a spectrum from a 
    shorter segment. The values are loaded from disk and decoded using
    the *values* method which takes a start time and either an end time
    or an integer number of points to include.
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
        self.samples = 0
        self.wavelength = 1550.0e-9  # this is bad! We should fix it!
        self.notes = dict()
        if self.digfile:
            self.load_dig_file()
        else:
            raise ValueError(f"I don't understand files with extension {ext}")

    @property
    def digfile(self):
        """Is this Spectrogram drawn from a .dig file?"""
        return self.ext == 'dig'

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
        top = [x.strip() for x in text[:headerLength//2].replace(
            "\000", "").split('\r\n') if x.strip()]
        self.decode_dig_header(top)

        bottom = [x.strip() for x in text[headerLength//2:].replace(
            "\000", "").split('\r\n') if x.strip()]
        self.headers = top

        # These fields are pretty short, but I think they're quite
        # descriptive. Do people think we should bulk them up?

        self.V0 = float(bottom[-1])  # voltage offset
        self.dV = float(bottom[-2])  # voltage interval
        self.t0 = float(bottom[-3])  # initial time
        self.dt = float(bottom[-4])  # sampling interval
        self.bits = int(bottom[-5])  # 8, 16, or 32
        self.samples = int(bottom[-6])
        self.bytes_per_point = self.bits // 8
        self.data_format = {1: np.uint8, 2: np.int16,
                            4: np.int32}[self.bytes_per_point]

        # We should pay attention to the endian-ness
        # We should probably adjust the datatype when we first decode the header,
        # rather than here. Can someone fix this?
        # Python stores the machine's endianess at sys.byteorder ('little'|'big')
        # The files we have seen so far are 'LSB', corresponding to 'little'
        # GEN3_CHANNEL1KEY001.dig header seems to have nothing about byte order
        if 'byte_order' in self.notes:
            native_order = 'MSB' if sys.byteorder == 'little' else 'LSB'
            if native_order != self.notes['byte_order']:
                self.data_format = np.dtype(
                    self.data_format).newbyteorder('S')  # swap the order

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
            f"{self.t0*1e6} µs to {(self.t0 + self.dt*self.samples)*1e6} µs in steps of {self.dt*1e12} ps"
        ])

    def __repr__(self):
        """ How we should represent a Spectrogram when we print it. 
        """

        output = "This spectrogram was generated from" + self.filename + ".\n"
        output += "The specs for the instrument were as follows: \n"
        output += self.notes.items().__repr__()
        return output


    def values(self, tStart, ending):
        """
        Return a numpy array with the properly normalized voltages corresponding to this segment.
        The ending argument can be an integer representing the number of points to include,
        or it can be a floating-point number indicating the ending time.
        """
        offset = 1024 if self.digfile else 0
        offset += self.bytes_per_point * self.point_number(tStart)
        if isinstance(ending, int):
            nSamples = ending
        else:
            nSamples = self.point_number(ending)
        with open(self.path, 'rb') as f:
            f.seek(offset)
            buff = f.read(nSamples * self.bytes_per_point)

        raw = np.frombuffer(buff, self.data_format, nSamples, 0)
        return raw * self.dV + self.V0

    def time_values(self, tStart, ending):
        """
        Return an array of time values corresponding to this interval. The arguments
        are the same as for the values method.
        """
        if isinstance(ending, int):
            nSamples = ending
            tFinal = tStart + (nSamples - 1) * self.dt
        else:
            tFinal = ending
            nSamples = 1 + int((tFinal - tStart) / self.dt)
        return np.linspace(tStart, tFinal, nSamples)

    def spectrum(self, t, nSamples, remove_dc=True):
        """
        Compute a spectrum from nSamples centered at time t
        """
        from spectrum import Spectrum
        tStart = t - nSamples // 2 * self.dt
        if tStart < self.t0:
            tStart = self.t0
        raw = self.values(tStart, nSamples)
        return Spectrum(raw, self.dt, remove_dc)

    def spectrogram(self, tStart, tEnd, fftSize=8192, floor=1e-2):
        """
        Compute a spectrogram. This needs work! There need to be
        lots more options that we either want to supply with
        default values or decode kwargs. But it should be a start.
        """
        freqs, times, spec = signal.spectrogram(
            self.values(tStart, tEnd),
            1.0 / self.dt,  # the sample frequency
            # ('tukey', 0.25),
            nperseg=fftSize,
            noverlap=None,
        )
        times += tStart
        # Convert to a logarithmic representation and use floor to attempt
        # to suppress some noise.
        spec = 20 * np.log10(spec + floor)
        # scale the frequency axis to velocity
        velocities = freqs * 0.5 * self.wavelength
        return {
            't': times,
            'v': velocities,
            'spectrogram': spec,
            'fftSize': fftSize,
            'floor': floor
            }

    def plot(self, sgram, **kwargs): # max_vel=6000, vmin=-200, vmax=100):
        plt.figure()
        plt.ylabel('Velocity (m/s)')
        plt.xlabel('Time (s)')
        if 'max_vel' in kwargs:
            plt.ylim(top=kwargs['max_vel'])
        options = {}
        for key, val in kwargs.items():
            if key == 'max_vel': continue
            options[key] = val
        plt.pcolormesh(sgram['t'], sgram['v'], sgram['spectrogram'], **options)
        plt.colorbar()



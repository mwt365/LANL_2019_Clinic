# coding:utf-8

"""
::
   Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
   Purpose: Compute a spectrogram from a DigFile
   Created: 9/20/19
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from ProcessingAlgorithms.preprocess.digfile import DigFile
from plotter import COLORMAPS

DEFMAP = '3w_gby'

class Spectrogram:
    """

    A Spectrogram takes a DigFile and a time range, as
    well as plenty of options, and generates a spectrogram
    using scipy.signal.spectrogram.

    Required arguments to the constructor:
        digfile: either an instance of DigFile or the filename of a .dig file

    **Optional arguments and their (default) values**

    t_start: (digfile.t0) time of the first point to use in the spectrogram
    ending:  (None) either the time of the last point or a positive integer
             representing the number of points to use; if None, the final
             point in the digfile is used.
    wavelength: (1550.0e-9) the wavelength in meters
    points_per_spectrum: (8192) the number of values used to generate
        each spectrum. Should be a power of 2.
    overlap: (1/4) the fraction of points_per_spectrum to overlap in
        successive spectra. An overlap of 0 means that each sample is used
        in only one spectrum. The default value means that successive
        spectra share 1/4 of their source samples.
    window_function: (None) the window function used by signal.spectrogram.
        The default value implies a ('tukey', 0.25) window.
    form: ('db') whether to use power values ('power'), decibels ('db'),
        or log(power) ('log') in reporting spectral intensities.
    convert_to_voltage: (True) scale the integer values stored in the
        .dig file to voltage before computing the spectrogram. If False,
        the raw integral values are used.
    detrend: ("linear") the background subtraction method.
    complex_value: (False) do you want to maintain the phase information as well. 
    **Computed fields**

    time:      array of times at which the spectra are computed
    frequency: array of frequencies present in each spectrum
    velocity:  array of velocities corresponding to each spectrum
    intensity: two-dimensional array of (scaled) intensities, which
               is the spectrogram. The first index corresponds to
               frequency/velocity, the second to time.
    """

    # The following fields may be set with the spectrogram.set call using
    # kwargs
    _fields = ("points_per_spectrum",
               "overlap",
               "window_function",
               "form",
               "use_voltage",
               "detrend")

    def __init__(self,
                 digfile: DigFile,
                 t_start=None,
                 ending=None,
                 wavelength: float = 1550.0e-9,
                 points_per_spectrum: int = 8192,
                 overlap: float = 0.25,
                 window_function=None,  # 'hanning',
                 form: str = 'db',
                 convert_to_voltage: bool = True,
                 detrend: str = "linear",
                 **kwargs
                 ):
        """
        Keyword arguments we handle:

        mode: if 'complex', compute the complex spectrogram, which gets
              stored in self.complex
        scaling: 'spectrum' or 'density'
        """
        if isinstance(digfile, str):
            digfile = DigFile(digfile)
        if isinstance(digfile, DigFile):
            self.data = digfile
        else:
            print(isinstance(digfile, DigFile))
            print(type(digfile))
            print(digfile)

            raise TypeError("Unknown file type")

        self.t_start = t_start if t_start != None else self.data.t0
        p_start, p_end = self.data._points(self.t_start, ending)
        self.t_end = self.t_start + self.data.dt * (p_end - p_start + 1)

        self.wavelength = wavelength
        self.points_per_spectrum = points_per_spectrum
        self.overlap = overlap
        self.window_function = window_function
        self.form = form
        self.use_voltage = convert_to_voltage
        self.detrend = detrend

        # the following will be set by _calculate
        self.time = None
        self.frequency = None
        self.velocity = None
        self.intensity = None
        # This will contain the intensity values if possible.
        # Otherwise it will contain the phase or angle information.
        # Containment determined by the value of computeMode.

        # This will only get set if self.complex_value is True
        self.orig_spec_output = None

        # deal with kwargs

        self.computeMode = "psd"  # This stands for power spectrum density.
        if "mode" in kwargs:
            available = ["psd", "complex", "magnitude", "angle", "phase"]
            desired = kwargs["mode"]
            if desired in available:
                self.computeMode = desired
            else:
                # Default to "all", but display error to the user. 
                # If the user specifies a mode that does not exist.
                kwargs["mode"] = "all"
                self.computeMode = "all"

        try:
            if False:
                self._load()
            else:
                raise Exception()
        except:
            self._compute(ending, **kwargs)
            # self._save()

    def _compute(self, ending, **kwargs):
        """
        Compute a spectrogram. This needs work! There need to be
        lots more options that we either want to supply with
        default values or decode kwargs. But it should be a start.
        """
        if self.use_voltage:
            vals = self.data.values(self.t_start, ending)
        else:
            vals = self.data.raw_values(self.t_start, ending)

        # possible modes are 'psd', 'complex', 'magnitude',
        # 'angle', and 'phase'
        mode = kwargs.get('mode', 'psd')
        scaling = kwargs.get('scaling', 'spectrum')
        if mode in ('angle', 'phase'):
            modes = [mode, 'psd']
        elif mode == "all":
            modes = ["complex", "magnitude", "angle", "phase", "psd"]
        else:
            modes = [mode]
        
        self.availableData = modes

        for mode in modes:
            freqs, times, spec = signal.spectrogram(
                vals,
                1.0 / self.data.dt,  # the sample frequency
                window=self.window_function if self.window_function else (
                    'tukey', 0.25),
                nperseg=self.points_per_spectrum,
                noverlap=int(self.overlap * self.points_per_spectrum),
                detrend=self.detrend,  # could be constant,
                scaling=scaling,
                mode=mode
            )
            setattr(self, mode, spec)
            self.orig_spec_output = spec


        times += self.t_start

        # Attempt to deduce baselines
        # baselines = np.sum(spec, axis=1)

        # Convert to a logarithmic representation and use floor to attempt
        # to suppress some noise.
        # I think the following is an attempt to normalize across
        # different numbers of points per spectrum.

        if mode == 'complex':
            self.complex = spec
            self.intensity = np.abs(spec)
            self.intensity *= self.intensity
        else:
            self.intensity = self.transform(spec)
        self.histogram_levels = self.histo_levels(self.intensity)

        # the first index is frequency, the second time
        self.frequency = freqs
        self.time = times

        # scale the frequency axis to velocity
        self.velocity = freqs * 0.5 * self.wavelength  # velocities

        # Now compute the probe destruction time.
        self.probeDestructionTime()
        
        self.estimatedStartTime_ = None
        # self.estimateStartTime()

    def transform(self, vals):
        """
        Perform any modification to values dictated by the value of self.form
        """
        epsilon = 1e-10
        if self.form == 'db':
            return 10 * np.log10(vals + epsilon) # Since you are already starting with power.
        if self.form == 'log':
            return np.log10(vals + epsilon)
        return vals

    def histo_levels(self, array: np.ndarray):
        """
        Sort a one-dimensional version of the array and report a vector of
        the values at 0%, 1%, ..., 100%
        """
        vals = array.flatten()
        vals.sort()
        indices = np.asarray(np.linspace(
            0, len(vals) - 1, 11), dtype=np.uint32)
        ladder = {'tens': vals[indices], }
        indices = np.linspace(indices[-2], indices[-1], 11, dtype=np.uint32)
        ladder['ones'] = vals[indices]
        indices = np.linspace(indices[-2], indices[-1], 11, dtype=np.uint32)
        ladder['tenths'] = vals[indices]
        return ladder

    def set(self, **kwargs):
        """
        Update the spectrogram to use the new parameters
        specified in the keyword arguments. If any changes cause
        the underlying values to change, recompute the spectrogram.
        """
        changed = False
        for field in self._fields:
            if field in kwargs and kwargs[field] != getattr(self, field):
                changed = True
                setattr(self, field, kwargs[field])
        if changed:
            self._compute(None)

    def __str__(self):
        return ", ".join(
            [str(x) for x in
             [self.data.filename,
              f"{self.points_per_spectrum} / {self.overlap*self.points_per_spectrum}",
              self.form,
              self.intensity.shape
              ]
             ])

    def _point_to_time(self, p: int):
        "Map a point index to a time"
        return self.time[p]

    def _time_to_index(self, t: float):
        "Map a time to a point number"
        p = (t - self.t_start) / (self.time[1] - self.time[0])
        p = int(0.5 + p)  # round to an integer
        if p < 0:
            return 0
        return min(p, len(self.time) - 1)

    def _velocity_to_index(self, v: float):
        "Map a velocity value to a point number"
        p = (v - self.velocity[0]) / (self.velocity[1] - self.velocity[0])
        p = int(0.5 + p)  # round
        if p < 0:
            return 0
        return min(p, -1 + len(self.velocity))

    def slice(self, time_range, velocity_range):
        """
        Input:
            time_range: Array/Tuple/List of times (t0, t1)
                t1 should be greater than t0 but we will handle the other case
            velocity_range: Array/Tuple/List of velocities (v0, v1)
                v1 should be greater than v0 but we will handle the other case
        Output:
            4 arrays time, velocity, intensity, original_spec
            time: the time values used in the measurement from t0 to t1 inclusive.
            velocity: the velocity values measured from v0 to v1 inclusive.
            intensity: the corresponding intensity values that we measured.
            original_spec: the output of the rolling FFT that we used. Depends on
                the value of self.computeMode
        """
        if time_range == None:
            time0, time1 = 0, len(self.time) - 1
        else:
            time0, time1 = [self._time_to_index(t) for t in time_range]
        if velocity_range == None:
            vel0, vel1 = 0, len(self.velocity) - 1
        else:
            vel0, vel1 = [self._velocity_to_index(v) for v in velocity_range]
        if time0 > time1:
            time0, time1 = time1, time0  # Then we will just swap them.
        if vel0 > vel1:
            # Then we will just swap them so that we can index normally.
            vel0, vel1 = vel1, vel0
        tvals = self.time[time0:time1 + 1]
        vvals = self.velocity[vel0:vel1 + 1]
        ivals = self.intensity[vel0:vel1 + 1, time0:time1 + 1]
        if self.computeMode != "psd":
            ovals = self.orig_spec_output[vel0:vel1 + 1, time0:time1 + 1]
            return tvals, vvals, ivals, ovals
        return tvals, vvals, ivals
    # Routines to archive the computed spectrogram and reload from disk

    def _location(self, location, create=False):
        """

        """
        if location == "":
            location = os.path.splitext(self.data.path)[0] + \
                '.spectrogram'
        if os.path.exists(location) and not os.path.isdir(location):
            raise FileExistsError
        if not os.path.exists(location) and create:
            os.mkdir(location)
        return location

    def _save(self, location=""):
        """
        Save a representation of this spectrogram.
        The format is a folder holding the three numpy arrays
        and a text file with the parameters.
        If the location is a blank string, the folder has
        the name of the digfile, with .dig replaced by .spectrogram.
        """
        location = self._location(location, True)
        with open(os.path.join(location, "properties"), 'w') as f:
            for field in self._fields:
                f.write(f"{field}\t{getattr(self,field)}\n")
        np.savez_compressed(
            os.path.join(location, "vals"),
            velocity=self.velocity,
            frequency=self.frequency,
            time=self.time,
            intensity=self.intensity)

    def _load(self, location=""):
        location = self._location(location)
        if not os.path.isdir(location):
            raise FileNotFoundError
        try:
            with open(os.path.join(
                    location, "properties"), 'r') as f:
                for line in f.readlines():
                    field, value = line.split('\t')
                    assert value == getattr(field)
            loaded = np.load(os.path.join(location, "vals"))
            for k, v in loaded.items():
                setattr(self, k, v)
            return True
        except Exception as eeps:
            print(eeps)
        return False

    @property
    def max(self):
        """The maximum intensity value"""
        return self.intensity.max()

    @property
    def min(self):
        """The minimum intensity value"""
        return self.intensity.min()

    @property
    def v_max(self):
        return self.wavelength * 0.25 / self.data.dt

    def power(self, values):
        """
        Given an np.array of intensity values from the spectrogram,
        return the corresponding power values (undoing any logarithms,
        if necessary).
        """
        if self.form == 'db':
            return np.power(10.0, 0.05 * values)
        if self.form == 'log':
            return np.power(10.0, values)
        return values

    def signal_to_noise(self):
        """
        Give an approximate signal to noise ratio for each time slice.
        Max/Mean
        """
        timeVelInten = np.transpose(self.intensity)
        answer = np.zeros(len(self.time))
        for ind in range(len(self.time)):
            answer[ind] = np.max(timeVelInten[ind]) / \
                np.mean(timeVelInten[ind])
        return answer

    def probeDestructionTime(self):
        """
        Compute an approximate value for the probe destruction time based
        upon the maximum total intensity for each time slice.
        """
        totalInten = np.sum(self.intensity, axis=0)
        maxInten = np.max(totalInten)
        inds = np.where(totalInten == maxInten)
        self.probe_destruction_time = self.time[inds][0]
        self.probe_destruction_index = inds[0][0]

        # Compute the maximum single intensity value and use that as another
        # estimate of probe destruction. Choose the smaller of the two options.
        a = self.max
        maxArray = np.max(self.intensity, axis=0)
        inds2 = np.where(maxArray == a)
        self.probe_destruction_index_max = inds2[0][0]
        if False:
            print("The value of PD index using the max estimate is", self.probe_destruction_index_max, "it has type", type(self.probe_destruction_index_max))
            print("The time array has shape", self.time.shape)

        self.probe_destruction_time_max = self.time[self.probe_destruction_index_max]

        import baselineTracking
        peaks, _, _ = baselineTracking.baselines.baselines_by_squash(self)
        self.probe_destruction_time_withBaselineTracking = baselineTracking.baselineTracking(self, peaks[0], 0.1, 20e-6)*1e-6

    def estimateStartTime(self):
        """
        Compute an approximate value for the jump off time based upon the change in 
        the baseline intensity.
        """
        import baselineTracking
        peaks, _, _ = baselineTracking.baselines.baselines_by_squash(self)
        self.estimatedStartTime_ = baselineTracking.baselineTracking(self, peaks[0], 0.024761904761904763)

    def plotHist(self, fig = None, minFrac=0.0, maxFrac = 1.0, numBins = 1001, **kwargs):
        if fig == None:
            fig = plt.figure()
        bins = np.linspace(self.min, self.max, numBins)

        threshold = bins[int(numBins*minFrac):int(numBins*maxFrac)]

        plt.hist(self.intensity.flatten(), threshold, **kwargs)
        axes = plt.gca()
        axes.set_ylabel('Counts', fontsize = 18)
        axes.set_xlabel('Intensity', fontsize = 18)
        axes.xaxis.set_tick_params(labelsize=14)
        axes.yaxis.set_tick_params(labelsize=14)        
        title = self.data.filename.split('/')[-1]
        axes.set_title(title.replace("_", "-")+" Intensity Histogram", fontsize = 24)
        
        return fig

    def plot(self, transformData = False, showBool:bool = True, **kwargs):
        # max_vel=6000, vmin=-200, vmax=100):
        pcms = {}
        if "psd" in self.availableData:
            self.availableData.append("intensity")
        if "complex" in self.availableData:
            self.availableData.append("real")
            self.availableData.append("imaginary")


        endTime = self._time_to_index((self.probe_destruction_time + self.probe_destruction_time_max)/2)
        # Our prediction for the probe destruction time. Just to make it easier to plot. 

        cmapUsed = COLORMAPS[DEFMAP]
        print(COLORMAPS.keys())
        if 'cmap' in kwargs:
            # To use the sciviscolor colormaps that we have downloaded.
            attempt = kwargs['cmap']
            if attempt in COLORMAPS.keys():
                cmapUsed = COLORMAPS[attempt]
                del kwargs['cmap']
        top = kwargs.get("max_vel", None)
        bot = kwargs.get("min_vel", None)
        right = kwargs.get("max_time", None)
        left = kwargs.get("min_time", None)

        print("The axes settings should be t,b,r,L", top, bot, right, left)

        if top != None:
            del kwargs['max_vel']
        if bot != None:
            del kwargs['min_vel']
        if left != None:
            del kwargs["min_time"]
        if right != None:
            del kwargs["max_time"]



        for data in self.availableData:
            zData = getattr(self, data, "psd") # getattr(object, itemname, default)
            if data == "complex":
                continue
            elif data == "real":
                zData = np.real(self.complex)
            elif data == "imaginary":
                zData = np.imag(self.complex)
            
            key = f"{data}" + (f" transformed to {self.form}" if transformData else " raw")
            fig = plt.figure(num=key)
            axes = plt.gca()

            pcm = None # To define the scope.
            if 'cmap' not in kwargs:
                pcm = axes.pcolormesh(
                    self.time[:endTime] * 1e6,
                    self.velocity,
                    self.transform(zData[:,:endTime]) if (data != "intensity" and transformData) else zData[:,:endTime],
                    cmap = cmapUsed,
                    **kwargs)
            else:
                pcm = axes.pcolormesh(
                    self.time[:endTime] * 1e6,
                    self.velocity,
                    self.transform(zData[:,:endTime]) if (data != "intensity" and transformData) else zData[:,:endTime],
                    **kwargs)

            if self.estimatedStartTime_ != None:
                # Plot the start time estimate.
                axes.plot([self.estimatedStartTime_]*len(self.velocity), self.velocity, "k-", label = "Estimated Start Time", alpha = 0.75)
                plt.legend()
            
            print(f"The current maximum of the colorbar is {np.max(zData[:,:endTime])} for the dataset {data}")
            plt.gcf().colorbar(pcm, ax=axes)
            axes.set_ylabel('Velocity (m/s)', fontsize = 14)
            axes.set_xlabel('Time ($\mu$s)', fontsize = 14)
            axes.xaxis.set_tick_params(labelsize=12)
            axes.yaxis.set_tick_params(labelsize=12)        
            title = self.data.filename.split('/')[-1]
            axes.set_title(title.replace("_", "-") + f" {data} spectrogram", fontsize = 24)

            axes.set_xlim(left, right)
            axes.set_ylim(bot, top) # The None value is the default value and does not update the axes limits.            

            pcms[key] = pcm

        if "complex" in self.availableData:
            self.availableData.remove("real")
            self.availableData.remove("imaginary")
        
        if showBool:
            plt.show() # Needed for Mac computers it seems.

        return pcms, axes


if __name__ == '__main__':
    # Then I am calling this from the command line and not jupyter. This is an assumption!
    if False:
        import tkinter as tk
        root = tk.Tk()
        width_px = root.winfo_screenwidth()
        height_px = root.winfo_screenheight()
        width_mm = root.winfo_screenmmwidth()
        height_mm = root.winfo_screenmmheight()
        # 2.54 cm = in
        width_in = width_mm / 25.4
        height_in = height_mm / 25.4

        # Set the default window size for matplotlib to be the fullscreen - 0.5 inches on the side and the top.

        plt.rcParams["figure.figsize"] = [width_in-0.5, height_in-0.5]

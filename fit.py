# coding:utf-8

"""
::

   Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
   Purpose: Manage curve fits with more bells and whistles
   Created: 12/16/19
"""
import re
import numpy as np
import inspect
from scipy.optimize import curve_fit, OptimizeWarning
from ProcessingAlgorithms.Fitting.moving_average import moving_average
import matplotlib.pyplot as plt
from scipy.stats import chi2


class Fit:
    """
    Object to perform a fit and compute residuals and chisq.

    If the fit is successful, the field Gaussian.valid is
    set to True.
    """

    def __init__(self, function, x: np.ndarray,
                 y: np.ndarray, p0: np.ndarray,
                 **kwargs):
        """
        kwargs may include hold="1011", where each 0 corresponds
        to a parameter allowed to vary and 1 to a value held fixed
        """
        self.x = x
        self.y = y
        self._f = function
        self.p0 = p0
        self.error = None
        self.errors = None
        self.sigma = None
        self.hold = kwargs.get("hold")
        self.dof = len(x) - len(p0)  # assumes nothing held
        # tex representation of variables
        self.tex_labels = kwargs.get('tex')
        self.chisq = None
        self.prob_greater = None

        try:

            if len(x) != len(y):
                raise f"x [{len(x)}] and y [{len(y)}] must have the same length"

            sigma = kwargs.get('sigma', None)
            if isinstance(sigma, (float, int)):
                sigma = np.ones(len(x)) * sigma
            self.sigma = sigma

            if self.hold:
                if not isinstance(self.hold, str):
                    raise "hold must be a string with 1 for fixed, 0 for variable"
                if len(self.hold) != len(self.p0):
                    raise "The number of digits in hold must match the number of parameters"
                # We need to copy the parameters held fixed and respect the ones being varied
                self.variable = [x for x in range(
                    len(p0)) if self.hold[x] == "0"]
                self.dof = len(x) - len(self.variable)

                def fhold(t, *params):
                    p = self.parameters(params)
                    return self._f(t, *p)
                f = fhold
                p0 = [self.p0[n] for n in self.variable]
            else:
                f = self._f

            try:
                self.params, self.covars = curve_fit(
                    f, x, y, p0=p0,
                    sigma=sigma, absolute_sigma=True)
            except OptimizeWarning:
                raise "Failed to converge"

            if np.inf in self.covars or np.nan in self.covars:
                self.error = "Failed to converge"
            else:
                # adjust for held parameters, if necessary
                self.params = self.parameters(self.params)
                self.residuals = self.y - self._f(self.x, *self.params)
                if isinstance(sigma, np.ndarray):
                    self.norm_residuals = self.residuals / sigma
                    self.chisq = np.sum(
                        self.norm_residuals ** 2)
                    self.reduced_chisq = self.chisq / self.dof
                    self.prob_greater = 1 - chi2.cdf(self.chisq, self.dof)
                errs = np.sqrt(np.diag(self.covars))
                if self.hold:
                    self.errors = np.zeros(self.params.size)
                    for n, pnum in enumerate(self.variable):
                        self.errors[pnum] = errs[n]
                else:
                    self.errors = errs

        except Exception as eeps:
            self.error = str(eeps)

    def __call__(self, t: np.ndarray):
        return self._f(t, *self.params)

    def __str__(self):
        name = self._f.__name__
        args = inspect.getfullargspec(self._f)
        argnames = args.args[1:]
        lines = [name, ]
        has_unc = isinstance(self.sigma, np.ndarray)
        if self.valid:
            for n in range(len(self.params)):
                lines.append(f"{argnames[n]:>16s} = {self.params[n]:^8.4g}")
                if has_unc:
                    lines[-1] += f" ± {self.errors[n]:.2g}"
                    if self.errors[n] > 0:
                        lines[-1] += f" ({abs(self.errors[n]/self.params[n])*100:.2g}%)"
            if has_unc:
                lines.append(f"N_dof = {self.dof}, chisq = {self.chisq:.3g}")
                lines[-1] += f" ({self.reduced_chisq:.3g})"
        else:
            lines.append(self.error)
        return "\n".join(lines)

    def texval(self, x):
        """
        Render a string representation of a value in TeX format
        """
        if "e" not in x:
            return x
        m = re.search(r'([-+0-9.]*)\s?e\s?([+-])\s?([0-9]*)', x)
        if m:
            s = m.group(1) + r" \times 10^{"
            if m.group(2) == '-':
                s += '-'
            s += str(int(m.group(3))) + "}"
            return s
        print("Ack! for " + x)
        return x

    def tex_val_unc(self, x, dx):
        """
        Produce a LaTeX representation of the value and its uncertainty
        """

        try:
            xdigits = int(np.log10(abs(x)))
            dxdigits = int(np.log10(dx))
            digits = 2 + xdigits - dxdigits
            round_spot = -xdigits + digits
            xround = round(x, round_spot)
            dxround = round(dx, round_spot)

            ratio = dx / abs(x)  # this will fail if dx is 0
            # digits = 2 + int(np.round(np.log10(ratio), 2))
            fmt = "{:0." + str(digits) + "g}"
            main = self.texval(fmt.format(xround))
            # To get the right number of digits, we need to
            # figure out the place of the LSD of x
            unc = self.texval(f"{dxround:.2g}")
            if ratio > 1:
                rel = f"{ratio:.2f}"
            else:
                rel = f"{100*ratio:.1f}"
                rel += "\\%"
            return (main, unc, rel)
        except Exception as eeps:
            print(eeps)
            return ("a", "b", "c")

    def tex(self):
        name = self._f.tex if hasattr(self._f, 'tex') else self._f.__name__
        if self.tex_labels:
            argnames = self.tex_labels
        else:
            args = inspect.getfullargspec(self._f)
            argnames = args.args[1:]
        lines = [name, ]
        TABLE = False

        has_unc = isinstance(self.sigma, np.ndarray)
        if self.valid:
            if TABLE:
                lines.append(r"\begin{tabular}{ccc}")
                lines.append(
                    r"\textbf{Param} & \textbf{Value} & \textbf{Unc.} & \textbf{Rel. Unc.} \\")
            for n in range(len(self.params)):
                if TABLE:
                    lines.append(argnames[n] + " & ")
                    lines.append(" & ".join(
                        self.tex_val_unc(self.params[n], self.errors[n])) +
                        r" \\")

                else:
                    lines.append(f"{argnames[n]:>16s} = $")
                    v, u, r = self.tex_val_unc(
                        self.params[n], self.errors[n])
                    vals = "{0} \\pm {1}\\; ({2})$".format(v, u, r)
                    lines[-1] += vals
                    # lines[-1] += self.texval(f"{self.params[n]:^8.4g}")
                    # if has_unc:
                    #    lines[-1] += "\\pm" + self.texval(f" {self.errors[n]:.2g}")
                    # if self.errors[n] > 0:
                    #    lines[-1] += f" ({abs(self.errors[n]/self.params[n])*100:.2g}\\%)"
                    # lines[-1] += "$"
            if TABLE:
                lines.append(r"\end{tabular}")
            if has_unc:
                lines.append(r"$N_{\rm dof} = " + self.texval(f"{self.dof}") + ", ")
                lines[-1] += r"\chi^2 = " + self.texval(f"{self.chisq:.3g}")
                lines[-1] += r"\; " + f" ({self.reduced_chisq:.3g})$"
                lines.append(r"$P_> = " + f"{100*self.prob_greater:.1f}" + "\%$")
        else:
            lines.append(self.error)
        return "\n".join(lines)

    def parameters(self, par):
        if self.hold:
            p = self.p0
            for n, pnum in enumerate(self.variable):
                p[pnum] = par[n]
            return p
        return par

    @property
    def valid(self):
        return self.error is None

    def plot(self, **kwargs):
        """

        """
        residuals, normalized_residuals = False, False
        title = kwargs.get('title', '')
        if self.valid:
            # See if we should plot residuals
            residuals = kwargs.get('residuals', True)
            normalized_residuals = kwargs.get('normalized_residuals', True)
        num_axes = 1 + (1 if residuals else 0) + \
            (1 if normalized_residuals else 0)
        gspec = dict(
            width_ratios=[1],
            height_ratios=[1],
            hspace=0.05,
            left=0.125,
            right=0.975
        )
        if num_axes == 2:
            gspec['height_ratios'] = [1, 4]
        elif num_axes == 3:
            gspec['height_ratios'] = [1, 1, 4]
        fig, axes = plt.subplots(
            nrows=num_axes, ncols=1,
            sharex=True, gridspec_kw=gspec,
            figsize=kwargs.get('figsize')
        )
        self.fig = fig
        if residuals:
            residuals = axes[0]
            if normalized_residuals:
                normalized_residuals = axes[1]
        elif normalized_residuals:
            normalized_residuals = axes[0]
        if num_axes == 1:
            main = axes
            main.set_title(title)
        else:
            main = axes[-1]
            axes[0].set_title(title)

        # Plot the data first, except lay down the fit curve
        # first
        logx = kwargs.get('logx', False)
        logy = kwargs.get('logy', False)
        xmin = kwargs.get('xmin', np.min(self.x))
        xmax = kwargs.get('xmax', np.max(self.x))
        ymin = kwargs.get('ymin', np.min(self.y))
        ymax = kwargs.get('ymax', np.max(self.y))
        xfit, yfit = kwargs.get('xfit'), kwargs.get('yfit')
        npoints = kwargs.get('npoints', 200)
        if logx:
            main.set_xscale('log', nonposx='clip')
            if not isinstance(xfit, np.ndarray):
                xfit = np.power(
                    10, np.linspace(
                        np.log10(xmin),
                        np.log10(xmax), npoints))
        elif not isinstance(xfit, np.ndarray):
            xfit = np.linspace(xmin, xmax, npoints)
        if not yfit:
            yfit = self.__call__(xfit)

        if logy:
            main.set_yscale('log', nonposy='clip')
        main.plot(xfit, yfit)
        main.errorbar(self.x, self.y, yerr=self.sigma,
                      fmt='.', alpha=0.5)

        # add an annotation
        legend = kwargs.get('legend', (0.1, 0.1))
        xpos = (1 - legend[0]) * xmin + legend[0] * xmax
        ypos = (1 - legend[1]) * ymin + legend[1] * ymax
        halign = 'left' if legend[0] < 0.3 else (
            'right' if legend[0] > 0.7 else 'center')
        valign = 'bottom' if legend[1] < 0.3 else (
            'top' if legend[1] > 0.7 else 'center')
        main.text(xpos, ypos, self.tex(),
                  horizontalalignment=halign,
                  verticalalignment=valign)
        if 'xlabel' in kwargs:
            main.set_xlabel(kwargs['xlabel'])
        if 'ylabel' in kwargs:
            main.set_ylabel(kwargs['ylabel'])

        if residuals:
            if logx:
                residuals.set_xscale('log', nonposx='clip')
            # plot the zero line
            residuals.plot([xmin, xmax], [0, 0], 'k-', alpha=0.5)
            residuals.errorbar(self.x, self.residuals,
                               yerr=self.sigma, fmt='.')
            residuals.set_ylabel('Res.')

        if normalized_residuals:
            if logx:
                normalized_residuals.set_xscale('log', nonposx='clip')
            # Compute colors
            normalized_residuals.plot([xmin, xmax], [0, 0], 'k-', alpha=0.5)
            colors = self.norm_res_colors
            normalized_residuals.scatter(
                self.x, self.norm_residuals,
                s=7**2, marker='o', c=colors)
            normalized_residuals.set_ylabel('N.R.')

    @property
    def norm_res_colors(self):
        sigmas = np.asarray(np.abs(self.norm_residuals), dtype=np.uint16)
        sigmas[sigmas > 3] = 3
        alpha = 0.75
        colormap = [
            (0, 0.7, 0, alpha),
            (0.9, 0.9, 0, alpha),
            (1.0, 0.7, 0, alpha),
            (1, 0, 0, alpha)]
        return np.array([colormap[x] for x in sigmas])


if __name__ == '__main__':
    from numpy.random import Generator, PCG64
    import matplotlib.pyplot as plt
    if False:
        rg = Generator(PCG64())
        x = np.linspace(-3, 3, 61)
        xfine = np.linspace(-3, 3, 601)
        center = rg.standard_normal()
        width = 0.1 + abs(0.1 * rg.standard_normal())
        amp = rg.uniform(0.1, 100)
        background = rg.uniform(0, 50)
        noise = rg.uniform(0.05 * amp, 0.25 * amp)
        arg = (x - center) / width
        y = background + amp * np.exp(-0.5 * arg**2)
        y += rg.normal(0, noise, len(y))
    else:
        with open('data.txt', 'r') as f:
            dat = []
            for line in f.readlines()[1:]:
                dat.append([float(x) for x in line.split('\t')])
            temp = np.asarray(dat)
        x = temp[:, 0]
        y = temp[:, 1]
        noise = sigma = temp[:, 2]
        amp = 75
        center = 100
        width = 12
        background = -10

    def gaussy(x, A, mu, sigma, background):
        return background + A * np.exp(-0.5 * ((x - mu) / sigma)**2)

    gus = Fit(gaussy, x, y, np.array([amp, width, center, background]),
              sigma=noise)  # , hold="1101")

    print("Fit results:")
    print(gus)
    print('\nActual parameters:')
    print(f'amplitude = {amp:.3g}')
    print(f'center = {center:.3g}  ({abs((center-gus.params[1])/gus.errors[1]):.1f} σ)')
    print(f'width = {width:.3g}  ({abs((width-abs(gus.params[2]))/gus.errors[2]):.1f} σ)')
    print(f'background = {background:.3g}')
    gus.plot(legend=(1, 1))
    plt.show()
    # plt.plot(x, y, 'ro', alpha=0.5)
    # plt.plot(x, moving_average(y), 'go', alpha=0.25)
    # xfine = np.linspace(x[0], x[-1], num=x.size * 10)
    # plt.plot(xfine, gus(xfine), 'k-')
    # plt.show()



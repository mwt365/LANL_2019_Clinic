# coding:utf-8

"""
::

   Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
   Purpose: Manage curve fits with more bells and whistles
   Created: 12/16/19
"""
import numpy as np
import inspect
from scipy.optimize import curve_fit, OptimizeWarning
from ProcessingAlgorithms.Fitting.moving_average import moving_average


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
    plt.plot(x, y, 'ro', alpha=0.5)
    plt.plot(x, moving_average(y), 'go', alpha=0.25)
    xfine = np.linspace(x[0], x[-1], num=x.size * 10)
    plt.plot(xfine, gus(xfine), 'k-')
    plt.show()



import re
from typing import List

import numpy as np
from scipy.sparse import csc_matrix, spdiags
from scipy.sparse.linalg import spsolve


def find_positive_peak_positions(data, threshold: float, min_dist: float) -> List[float]:
    '''
    Finds positive peak positions with a height threshold and a minimum distance.

    :param data: datalist
    :param threshold: minimum height for a peak
    :param min_dist: minimum distance between peaks

    :return: A list of peak positions
    '''
    x, y = data
    gradients = np.diff(y)
    peaks = []
    for i, gradient in enumerate(gradients[:-1]):
        if (gradients[i] > 0) & (gradients[i + 1] <= 0) & (y[i] > threshold):
            if len(peaks) > 0:
                if (x[i + 1] - peaks[-1]) > min_dist:
                    peaks.append(x[i + 1])
            else:
                peaks.append(x[i + 1])
    return np.array(peaks)


def find_positive_peaks(datalist, threshold, min_dist):
    peak_positions = []
    for i, data in enumerate(datalist):
        peak_pos = find_positive_peak_positions(data, threshold=threshold, min_dist=min_dist)
        peak_positions.append(peak_pos)
    return peak_positions


def gaussian(x, area: float, position: float, fwhm: float):  # Equation by Urmas
    return area / (fwhm * np.sqrt(np.pi / (4 * np.log(2)))) * np.exp(-4 * np.log(2) * (x - position) ** 2 / fwhm ** 2)


def gaussian_width_fwhm(x, position: float, height: float, fwhm: float):
    return height * np.exp(-4 * np.log(2) * (x - position) ** 2 / fwhm ** 2)


def multiple_gaussian_with_baseline_correction(x, *args):
    '''
    :param x: x-axis for the gaussian
    :param args: First three args are constant, linear and quadratic coefficients for the baseline
                 After that comes position, area, fwhm for every peak
    :return: y values for the gaussians and the baseline
    '''
    constant, linear, quadratic = args[:3]
    y = constant + linear * x + quadratic * (x - (min(x) + max(x)) / 2) ** 2
    number_of_peaks = int((len(args) - 3) / 3)
    for i in range(number_of_peaks):
        position, area, fwhm = args[i * 3 + 3:i * 3 + 6]
        y += gaussian(x, area, position, fwhm)
    return y


def baseline_als(y, lam=30000, p=0.005, niter=10):
    ## Eilers baseline correction for Asymmetric Least Squares
    ## Migrated from MATLAB original by Kristian Hovde Liland
    ## $Id: baseline.als.R 170 2011-01-03 20:38:25Z bhm $
    #
    # INPUT:
    # spectra - rows of spectra
    # lambda  - 2nd derivative constraint - smoothness
    # p       - regression weight for positive residuals - asymmetry
    # niter   - max internations count
    # VARIABLES:
    # w      - final regression weights
    # corrected - baseline corrected spectra
    # OUTPUT:
    # z  - proposed baseline

    L = len(y)
    D = csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)

    for _ in range(niter):
        W = spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        # Restricted regression
        z = spsolve(Z, w * y)
        # Weights for next regression
        w = p * (y > z) + (1 - p) * (y < z)
    #  return z, Z, w, W
    return z


def plot_waterfall(datalist, parameters, ax, **kwargs):
    color = kwargs.get('line_color', 'black')
    lines = []
    for i, data in enumerate(datalist):
        lines.append(ax.plot(data[0], data[1] + parameters[i] * kwargs['stackscale'], color)[0])
    return lines


def trim_spectra(spectra, start=None, end=None):
    if start is not None:
        smaller_than_start = np.where(spectra[0] < start)
        spectra = np.delete(spectra, smaller_than_start, axis=1)
    if end is not None:
        larger_than_end = np.where(spectra[0] > end)
        spectra = np.delete(spectra, larger_than_end, axis=1)
    return spectra


def find_num_from_name(fname, decimal_character, delimiter):
    ''' Returns the first number

    The returnable element is the first occuring number with decimal_character
    as decimal and surrounded by delimiters
    '''
    needs_escaping = ['.']
    if decimal_character not in needs_escaping:
        regex = '{0}[-]?[0-9]*{1}[0-9]*{0}'.format(delimiter, decimal_character)
    else:
        regex = '{0}[-]?[0-9]*\{1}[0-9]*{0}'.format(delimiter, decimal_character)
    value = re.search(regex, fname).group()
    value = float(value.replace(delimiter, '').replace(decimal_character, "."))
    return value

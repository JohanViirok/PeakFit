import sys
from typing import List
import os
import copy

import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QSlider, QLineEdit, QLabel, QHBoxLayout, QVBoxLayout, \
    QFileDialog, QProgressBar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import RectangleSelector
from scipy import optimize

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
    return area / (fwhm * np.sqrt(np.pi * np.log(2))) * np.exp(-4 * np.log(2) * (x - position) ** 2 / fwhm ** 2)


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

class FitResultPlot(QDialog):
    def __init__(self, fitted_parameters, fit_errors, parameters, xlimits):
        super(FitResultPlot, self).__init__()
        self.fitted_parameters = fitted_parameters
        self.fit_errors = fit_errors
        # a figure instance to plot on
        self.figure = plt.figure(figsize=(10, 10))
        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax = self.figure.add_subplot(111)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.plot_fitted_parameters(parameters)
        self.ax.set_xlim(xlimits)
        self.setLayout(layout)

    def plot_fitted_parameters(self, parameters):
        for i, fit_parameters in enumerate(self.fitted_parameters):
            number_of_peaks = int((len(fit_parameters) - 3) / 3)
            peaks = fit_parameters[3::3]
            fields = [parameters[i]] * number_of_peaks
            areas = fit_parameters[4::3]
            xerr = self.fit_errors[i][3::3]
            self.ax.scatter(peaks, fields, areas)
            self.ax.errorbar(peaks, fields, xerr=xerr, ecolor='red', fmt='None')
            self.ax.set_ylabel('Magnetic Field (T)')
            self.ax.set_xlabel('Wavenumber (cm$^{-1}$)')
            self.figure.tight_layout()



class Window(QDialog):
    def __init__(self, datalist, parameters, parent=None):
        super(Window, self).__init__(parent)
        self.parameters = parameters

        for i, data in enumerate(datalist):
            datalist[i] = np.array([data[0][np.isfinite(data[1])], data[1][np.isfinite(data[1])]])
        self.original_datalist = copy.deepcopy(datalist)
        self.datalist = copy.deepcopy(datalist)
        self.fitted_peak_parameters = []
        self.fit_errors = []

        self.peaks = [[] for _ in range(len(datalist))]
        self.fit_lines = []
        self.als_baselines = []

        self.resize(1440, 900)
        self.setWindowFlags(self.windowFlags() |
                            Qt.WindowSystemMenuHint |
                            Qt.WindowMinMaxButtonsHint)
        self.setWindowTitle('FitPeaks')

        # a figure instance to plot on
        self.figure = plt.figure(figsize=(10, 10))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Box for buttons
        self.resetButton = QPushButton('Reset plot')
        self.resetButton.clicked.connect(self.reset_plot)

        # Limits
        self.xlimitsLabel = QLabel(self)
        self.xlimitsLabel.setText('x limits:')
        self.xlimitsMinValueBox = QLineEdit(self)
        self.xlimitsMinValueBox.setFixedWidth(50)
        self.xlimitsMinValueBox.editingFinished.connect(self.xlimit_changed)
        self.xlimitsMinValueBox.setValidator(QDoubleValidator(0, 250, 2))
        self.xlimitsMaxValueBox = QLineEdit(self)
        self.xlimitsMaxValueBox.setFixedWidth(50)
        self.xlimitsMaxValueBox.editingFinished.connect(self.xlimit_changed)
        self.xlimitsMaxValueBox.setValidator(QDoubleValidator(0, 250, 2))

        # Stack
        self.stackscaleSlider = QSlider(Qt.Horizontal)
        self.stackscaleSlider.setValue(20)
        self.stackscaleSlider.setMinimum(-100)
        self.stackscaleSlider.setMaximum(100)
        self.stackscaleSlider.setTickInterval(100)
        self.stackscaleSlider.setTickPosition(QSlider.TicksBothSides)
        self.stackscaleSlider.valueChanged.connect(self.stackscale_slider_changed)
        self.stackscaleLabel = QLabel(self)
        self.stackscaleLabel.setText('Stack:')
        self.stackscaleValueBox = QLineEdit(self)
        self.stackscaleValueBox.setFixedWidth(50)
        self.stackscaleValueBox.setText(str(self.stackscaleSlider.value()))
        self.stackscaleValueBox.editingFinished.connect(self.stackscale_valuebox_changed)
        self.stackscaleBox = QHBoxLayout()
        self.stackscaleBox.addWidget(self.xlimitsLabel)
        self.stackscaleBox.addWidget(self.xlimitsMinValueBox)
        self.stackscaleBox.addWidget(self.xlimitsMaxValueBox)
        self.stackscaleBox.addWidget(self.stackscaleLabel)
        self.stackscaleBox.addWidget(self.stackscaleValueBox)
        self.stackscaleBox.addWidget(self.stackscaleSlider)
        self.stackscale = self.stackscaleSlider.value()

        # Threshold
        self.thresholdSlider = QSlider(Qt.Horizontal)
        self.thresholdSlider.setFocusPolicy(Qt.StrongFocus)
        self.thresholdSlider.setValue(70)
        self.thresholdSlider.setMaximum(500)
        self.thresholdSlider.setTickInterval(1)
        self.thresholdSlider.setSingleStep(1)
        self.thresholdSlider.valueChanged.connect(self.find_peaks)
        self.thresholdSlider.valueChanged.connect(self.change_threshold)
        self.thresholdLabel = QLabel(self)
        self.thresholdLabel.setText('Threshold:')
        self.thresholdValueBox = QLineEdit(self)
        self.thresholdValueBox.setFixedWidth(50)
        self.thresholdValueBox.setText(str(self.thresholdSlider.value() / 10))
        self.thresholdBox = QHBoxLayout()
        self.thresholdBox.addWidget(self.thresholdLabel)
        self.thresholdBox.addWidget(self.thresholdValueBox)
        self.thresholdBox.addWidget(self.thresholdSlider)

        # ALS Baseline Lambda
        self.ALSLambdaSlider = QSlider(Qt.Horizontal)
        self.ALSLambdaSlider.setFocusPolicy(Qt.StrongFocus)
        self.ALSLambdaSlider.setValue(300000)
        self.ALSLambdaSlider.setMaximum(1000000)
        self.ALSLambdaSlider.setTickInterval(1)
        self.ALSLambdaSlider.setSingleStep(1)
        self.ALSLambdaSlider.sliderReleased.connect(self.calculate_als_baseline)
        self.ALSLambdaLabel = QLabel(self)
        self.ALSLambdaLabel.setText('Lambda:')
        self.ALSLambdaValueBox = QLineEdit(self)
        self.ALSLambdaValueBox.setFixedWidth(50)
        self.ALSLambdaValueBox.setText(str(self.ALSLambdaSlider.value() / 10))
        self.ALSLambdaBox = QHBoxLayout()
        self.ALSLambdaBox.addWidget(self.ALSLambdaLabel)
        self.ALSLambdaBox.addWidget(self.ALSLambdaValueBox)
        self.ALSLambdaBox.addWidget(self.ALSLambdaSlider)

        # ALS Baseline Positive weight
        self.ALSPositiveWeightSlider = QSlider(Qt.Horizontal)
        self.ALSPositiveWeightSlider.setFocusPolicy(Qt.StrongFocus)
        self.ALSPositiveWeightSlider.setValue(500)
        self.ALSPositiveWeightSlider.setMaximum(3000)
        self.ALSPositiveWeightSlider.setTickInterval(1)
        self.ALSPositiveWeightSlider.setSingleStep(1)
        self.ALSPositiveWeightSlider.sliderReleased.connect(self.calculate_als_baseline)
        self.ALSPositiveWeightLabel = QLabel(self)
        self.ALSPositiveWeightLabel.setText('Positive weight:')
        self.ALSPositiveWeightValueBox = QLineEdit(self)
        self.ALSPositiveWeightValueBox.setFixedWidth(50)
        self.ALSPositiveWeightValueBox.setText(str(self.ALSPositiveWeightSlider.value() / 100000))
        self.ALSPositiveWeightBox = QHBoxLayout()
        self.ALSPositiveWeightBox.addWidget(self.ALSPositiveWeightLabel)
        self.ALSPositiveWeightBox.addWidget(self.ALSPositiveWeightValueBox)
        self.ALSPositiveWeightBox.addWidget(self.ALSPositiveWeightSlider)

        # Minimum distance
        self.minDistSlider = QSlider(Qt.Horizontal)
        self.minDistSlider.setFocusPolicy(Qt.StrongFocus)
        self.minDistSlider.setValue(2)
        self.minDistSlider.setMaximum(30)
        self.minDistSlider.setTickInterval(1)
        self.minDistSlider.setSingleStep(1)
        # self.minDistSlider.valueChanged.connect(self.find_peaks)
        self.minDistSlider.valueChanged.connect(self.change_min_distance)
        self.minDistLabel = QLabel(self)
        self.minDistLabel.setText('Minimum distance:')
        self.minDistValueBox = QLineEdit(self)
        self.minDistValueBox.setText(str(self.minDistSlider.value()))
        self.minDistValueBox.setFixedWidth(50)
        self.minDistBox = QHBoxLayout()
        self.minDistBox.addWidget(self.minDistLabel)
        self.minDistBox.addWidget(self.minDistValueBox)
        self.minDistBox.addWidget(self.minDistSlider)

        # Find peaks
        self.findPeaksButton = QPushButton('Find peaks')
        self.findPeaksButton.clicked.connect(self.find_peaks)

        # Remove peaks
        self.removePeaksButton = QPushButton('Remove peaks')
        self.removePeaksButton.clicked.connect(self.remove_all_peaks)

        # Subtract lambda
        self.subtractALSButton = QPushButton('Subtract ALS')
        self.subtractALSButton.clicked.connect(self.subtract_als_from_datalist)

        # Fit peaks
        self.fitPeaksButton = QPushButton('Fit peaks')
        self.fitPeaksButton.clicked.connect(self.fit_peaks)

        # Save fit results
        self.saveFitResultsButton = QPushButton('Save fit results')
        self.saveFitResultsButton.clicked.connect(self.save_fit_results)
        self.saveFitResultsButton.setEnabled(False)

        # Show fit plot
        self.fitPlotButton = QPushButton('Show fit plot')
        self.fitPlotButton.clicked.connect(self.show_fit_result_plot)
        self.fitPlotButton.setEnabled(False)

        # Progress bar
        self.progressBar = QProgressBar(self)
        self.progressBar.setMaximum(len(self.datalist))
        self.progressBar.setFormat('%v')
        self.progressBar.hide()

        self.plotButtonBox = QHBoxLayout()
        self.plotButtonBox.addWidget(self.resetButton)
        self.plotButtonBox.addWidget(self.findPeaksButton)
        self.plotButtonBox.addWidget(self.removePeaksButton)
        self.plotButtonBox.addWidget(self.subtractALSButton)
        self.plotButtonBox.addWidget(self.fitPeaksButton)
        self.plotButtonBox.addWidget(self.progressBar)
        self.plotButtonBox.addStretch(1)
        self.plotButtonBox.addWidget(self.fitPlotButton)
        self.plotButtonBox.addWidget(self.saveFitResultsButton)


        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addLayout(self.plotButtonBox)
        layout.addLayout(self.stackscaleBox)
        layout.addLayout(self.thresholdBox)
        layout.addLayout(self.minDistBox)
        layout.addLayout(self.ALSLambdaBox)
        layout.addLayout(self.ALSPositiveWeightBox)

        self.setLayout(layout)
        self.plot()
        # self.find_peaks()

    def calculate_als_baseline(self):
        lambda_ = self.ALSLambdaSlider.value() / 10
        positive_weight = self.ALSPositiveWeightSlider.value() / 100000
        self.ALSPositiveWeightValueBox.setText(str(positive_weight))
        self.ALSLambdaValueBox.setText(str(lambda_))
        als_exists = len(self.als_baselines) > 0
        lines, als_baselines = [], []

        for i, data in enumerate(self.datalist):
            als_baselines.append(np.array([data[0], baseline_als(data[1], lambda_, positive_weight)]))
            if als_exists:
                self.als_baseline_lines[i].set_ydata(als_baselines[i][1] + self.parameters[i] * self.stackscale)
            else:
                line = self.ax.plot(als_baselines[i][0], als_baselines[i][1] + self.parameters[i] * self.stackscale, 'g')
                lines.extend(line)
                self.als_baseline_lines = lines
        self.als_baselines = als_baselines
        self.canvas.draw()

    def subtract_als_from_datalist(self):
        for i,data in enumerate(self.datalist):
            self.datalist[i][1] = data[1] - self.als_baselines[i][1]
        self.subtractALSButton.setDisabled(True)
        self.als_baselines = []
        self.als_baseline_lines = []
        self.plot()


    def plot(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.als_baselines = []
        self.als_baseline_lines = []
        self.fit_lines = []
        # self.trim_spectra()

        self.lines = plot_waterfall(self.datalist, self.parameters, self.ax, stackscale=self.stackscale, line_color='red')
        self.plot_peak_positions()
        self.ax.set_xlabel('Wavenumber (cm$^{-1}$)')

        self.rectangle_selector = RectangleSelector(self.ax, self.remove_peaks, drawtype='box', useblit=True,
                                                    button=[1, 3],  # don't use middle button
                                                    minspanx=5, minspany=5, spancoords='pixels', interactive=False)
        cid = self.figure.canvas.mpl_connect('button_press_event', self.onclick)
        self.change_xlimit()
        self.canvas.draw()

    def change_xlimit(self):
        xmin, xmax = self.get_xlimits()
        self.ax.set_xlim((xmin, xmax))

    def get_xlimits(self):
        try:
            xmin = float(self.xlimitsMinValueBox.text())
        except ValueError:
            xmin = np.min([np.min(x[0]) for x in self.datalist])
        try:
            xmax = float(self.xlimitsMaxValueBox.text())
        except ValueError:
            xmax = np.max([np.max(x[0]) for x in self.datalist])
        return xmin, xmax

    def autoscale_y_axis(self):
        if self.stackscaleSlider.value() >= 0:
            ymin = np.min([np.min(x[1]) for x in self.datalist])
            ymax = np.max([np.max(x[1])+self.parameters[i]*self.stackscale for i,x in enumerate(self.datalist)])
        else:
            ymax = np.max([np.max(x[1]) for x in self.datalist])
            ymin = np.min([np.min(x[1])+self.parameters[i]*self.stackscale for i,x in enumerate(self.datalist)])
        delta = abs(ymax-ymin)*0.02
        self.ax.set_ylim((ymin-delta, ymax+delta))


    def stackscale_valuebox_changed(self):
        self.stackscaleSlider.setValue(int(self.stackscaleValueBox.text()))
        self.change_stackscale()

    def stackscale_slider_changed(self):
        self.stackscaleValueBox.setText(str(self.stackscaleSlider.value()))
        self.change_stackscale()

    def change_stackscale(self):
        self.stackscale = self.stackscaleSlider.value()
        for i,line in enumerate(self.lines):
            line.set_ydata(self.datalist[i][1] + self.stackscale * self.parameters[i])
        self.plot_peak_positions()
        if len(self.fit_lines) > 0:
            self.plot_fitted_data()
        for i,line in enumerate(self.als_baseline_lines):
            line.set_ydata(self.als_baselines[i][1] + self.stackscale*self.parameters[i])
        self.autoscale_y_axis()
        self.canvas.draw()


    def change_threshold(self):
        self.thresholdValueBox.setText(str(self.thresholdSlider.value() / 10))
        self.find_peaks()

    def change_min_distance(self):
        self.minDistValueBox.setText(str(self.minDistSlider.value()))

    def find_peaks(self):

        peaks = find_positive_peaks(self.datalist, threshold=float(self.thresholdValueBox.text()),
                                    min_dist=self.minDistSlider.value())
        self.peaks = peaks
        self.plot_peak_positions()

    def plot_peak_positions(self):
        try:
            self.scatter_points.remove()
        except (AttributeError, ValueError):
            pass

        xmin, xmax = self.get_xlimits()
        peak_xy = [[], []]
        for i, peaks_per_parameter in enumerate(self.peaks):
            if peaks_per_parameter is not None:
                peaks_in_range = [x for x in peaks_per_parameter if xmin <= x <= xmax]
                peak_xy[0].extend(peaks_in_range)
                peak_xy[1].extend(len(peaks_in_range) * [self.parameters[i] * self.stackscale])
        self.scatter_points = self.ax.scatter(peak_xy[0], peak_xy[1], color='k')
        self.canvas.draw()

    def fit_peaks(self):
        self.progressBar.show()
        peaks = self.peaks
        fit_data, lines, fit_errors = [], [], []
        use_previous_fit_results = len(self.fitted_peak_parameters) > 0
        use_previous_fit_results = False
        als_baseline_calculated = len(self.als_baselines) > 0

        for i, data in enumerate(self.datalist):
            x, y = data
            if als_baseline_calculated:
                y = y - self.als_baselines[i][1]
            initial_parameters = [0, 0, 0]
            bounds_min = [-np.inf,-np.inf,-np.inf]
            bounds_max = [np.inf, np.inf, np.inf]
            if peaks[i] is not None:
                for peak in peaks[i]:
                    initial_parameters.extend([float(peak), 10, 1])
                    bounds_min.extend([peak-3, 0, 0])
                    bounds_max.extend([peak+3, np.inf, np.inf])
            if use_previous_fit_results:
                number_of_peaks_difference = int(len(peaks[i]) - (len(self.fitted_peak_parameters[i])-3)/3)
                initial_parameters = self.fitted_peak_parameters[i]
                if number_of_peaks_difference > 0:
                    # only_one_peak = 1 if len(peaks[i]) == 1 else 0
                    print(initial_parameters, peaks[i])
                    start_index = int(len(peaks[i]) - number_of_peaks_difference)
                    for j in range(start_index, len(peaks[i])):
                        print('adding one peak to parameters')
                        initial_parameters = np.append(initial_parameters, [float(peaks[i][j]), 10, 1])
                    print(initial_parameters)
                    print(bounds_min)
                    print(bounds_max)
            try:
                fitted_parameters, covariance = optimize.curve_fit(multiple_gaussian_with_baseline_correction, x, y,
                                                                   p0=initial_parameters, bounds=(bounds_min,bounds_max))
            except RuntimeError:
                print("ERROR: Couldn't fit parameter: {}".format(self.parameters[i]))
                fitted_parameters, covariance = [0, 0, 0], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            fit_errors.append(np.sqrt(np.diag(covariance)))
            fit_data.append(fitted_parameters)
            print('Finished fitting parameter: {}'.format(self.parameters[i]))
            self.progressBar.setValue(i)
        print('*'*5 + '\t Fitting finished\t' + '*'*5)
        self.progressBar.hide()

        self.fitted_peak_parameters = fit_data
        self.fit_errors = fit_errors
        self.plot_fitted_data()
        self.saveFitResultsButton.setEnabled(True)
        self.fitPlotButton.setEnabled(True)

        self.canvas.draw()

    def plot_fitted_data(self):
        lines_exist_already = len(self.fit_lines) > 0
        als_baseline_calculated = len(self.als_baselines) > 0
        lines = []

        xmin, xmax = self.get_xlimits()
        x = np.linspace(xmin, xmax, 500)

        if len(self.fitted_peak_parameters) > 0:
            for i,data in enumerate(self.datalist):
                als_baseline = self.als_baselines[i][1] if als_baseline_calculated else 0
                if lines_exist_already:
                    fit_line = multiple_gaussian_with_baseline_correction(x, *self.fitted_peak_parameters[i])
                    self.fit_lines[i].set_ydata(fit_line + self.parameters[i] * self.stackscale + als_baseline)
                else:
                    line = self.ax.plot(x, multiple_gaussian_with_baseline_correction(x, *self.fitted_peak_parameters[i]) +
                                        self.parameters[i] * self.stackscale + als_baseline, 'b')
                    lines.extend(line)
            if not lines_exist_already:
                self.fit_lines = lines
        self.canvas.draw()


    def save_fit_results(self):
        num_of_peaks = [int((len(x) - 3) / 3) for x in self.fitted_peak_parameters]
        fileName, _ = QFileDialog.getSaveFileName(self, 'Save fit results', os.path.dirname(os.path.realpath(__file__)),
                                                  "Dat Files (*.dat);;All Files (*)")
        with open(fileName, 'w+') as f:
            comment = '### ALS Lambda: {}, ALS Positive Weight{}: \n'.format(self.ALSLambdaValueBox.text(),
                                                                             self.ALSPositiveWeightValueBox.text())
            labels = ['Parameter', 'constant', '+- constant', 'linear', '+- linear', 'quadratic', '+- quadratic']
            labels.extend(['position', '+- position', 'area', '+- area', 'fwhm', '+- fwhm'] * max(num_of_peaks))
            f.write(comment + '\t'.join(labels))

            for i, parameters in enumerate(self.fitted_peak_parameters):
                field = self.parameters[i]
                row = '\t'.join([('\t'.join((str(x), str(self.fit_errors[i][:][j])))) for j, x in
                                 enumerate(self.fitted_peak_parameters[i][:])])
                f.write('\t'.join(('\n' + str(field), row)))

    def show_fit_result_plot(self):
        self.FitResultPlot = FitResultPlot(self.fitted_peak_parameters, self.fit_errors, self.parameters, self.ax.get_xlim())
        self.FitResultPlot.show()

    def onclick(self, event):
        if event.dblclick:
            index = (np.abs(self.parameters - event.ydata / self.stackscale)).argmin()

            self.peaks[index] = np.append(self.peaks[index], event.xdata)
            self.plot_peak_positions()

            print(self.peaks[index])

    def remove_peaks(self, eclick, erelease):
        '''
        Removes peak positions in selected area
        :param eclick: mouse press event
        :param erelease: mouse release event
        '''
        x = eclick.xdata, erelease.xdata
        y = eclick.ydata, erelease.ydata
        for i in range(len(self.peaks)):
            if min(y) < self.parameters[i] * self.stackscale < max(y):
                if len(self.fitted_peak_parameters) > 0:
                    remove = [list(self.peaks[i]).index(p) for p in self.peaks[i] if min(x) < p < max(x)]
                    remove = [[x*3+3, x*3+4, x*3+5] for x in remove]
                    self.fitted_peak_parameters[i] = np.delete(self.fitted_peak_parameters[i], remove)
                self.peaks[i] = [p for p in self.peaks[i] if not min(x) < p < max(x)]
        self.plot_peak_positions()


    def remove_all_peaks(self):
        self.peaks = [[] for _ in range(len(self.datalist))]
        self.plot_peak_positions()

    def xlimit_changed(self):
        self.trim_spectra()
        self.plot()

    def trim_spectra(self):
        start = float(self.xlimitsMinValueBox.text()) if self.xlimitsMinValueBox.text() != '' else None
        end = float(self.xlimitsMaxValueBox.text()) if self.xlimitsMaxValueBox.text() != '' else None
        for i, data in enumerate(self.original_datalist):
            self.datalist[i] = trim_spectra(data, start=start, end=end)

    def reset_plot(self):
        self.datalist = copy.deepcopy(self.original_datalist)
        self.trim_spectra()
        self.fitted_peak_parameters = []
        self.fit_lines = []
        self.fit_errors = []
        self.plot()
        self.plot_peak_positions()
        self.plot_fitted_data()
        self.subtractALSButton.setEnabled(True)

def run(datalist, parameters):
    app = 0  # Fix for spyder crashing on consequent runs
    app = QApplication(sys.argv)
    main = Window(datalist, parameters)
    main.show()
    sys.exit(app.exec_())
	
import copy
import os
import re
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QDoubleValidator, QColor, QIcon
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QSlider, QLineEdit, QLabel, QHBoxLayout, QVBoxLayout, \
    QFileDialog, QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import RectangleSelector
from scipy import optimize
from scipy.sparse import csc_matrix, spdiags
from scipy.sparse.linalg import spsolve

INITIAL_PEAK_AREA = 10
INITIAL_PEAK_FWHM = 1
ALLOW_NEGATIVE_PEAKS = True


class MyToolbar(NavigationToolbar):
    def __init__(self, figure_canvas, parent= None):
        self.Window = parent
        self.toolitems = (
              ('Home', 'Reset original view', 'home', 'home'),
              ('Back', 'Back to  previous view', 'back', 'back'),
              ('Forward', 'Forward to next view', 'forward', 'forward'),
              (None, None, None, None),
              ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
              ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
              ('Subplots', 'Configure subplots', 'subplots', 'configure_subplots'),
              (None, None, None, None),
              ('Save', 'Save the figure', 'filesave', 'save_figure'),
              (None, None, None, None),
              ('Add Peak', 'Add peak manually', 'add_peak', 'add_peak_tool'),
            )
        NavigationToolbar.__init__(self, figure_canvas, parent= None)
        self._actions['add_peak_tool'].setCheckable(True)

    def add_peak_tool(self):
        if self._active == 'ADD':
            self._active = None
        else:
            self._active = 'ADD'
        if self._idPress is not None:
            self._idPress = self.canvas.mpl_disconnect(self._idPress)
            self.mode = ''

        if self._idRelease is not None:
            self._idRelease = self.canvas.mpl_disconnect(self._idRelease)
            self.mode = ''
        if self._active:
            self._idPress = self.canvas.mpl_connect('button_press_event', self.press_add_peak)
            self._idRelease = self.canvas.mpl_connect('button_release_event', self.release_add_peak)
            self.mode = 'add_peak_tool'
            self.canvas.widgetlock(self)
        else:
            self.canvas.widgetlock.release(self)

        self.set_message(self.mode)
        self._update_buttons_checked()

    def _update_buttons_checked(self):
        super(MyToolbar, self)._update_buttons_checked()
        self._actions['add_peak_tool'].setChecked(self._active == 'ADD')

    def press_add_peak(self, event):
        if event.button == 1:
            if event.inaxes is not None:
                self.add_peak_event = event
                x = np.linspace(*self.canvas.figure.axes[0].get_xlim(), 400)
                self.draw_fit_line, = self.canvas.figure.axes[0].plot(x, gaussian_width_fwhm(x, event.xdata, 0, 0) + event.ydata*self.Window.stackscale, c='k')
                self.peak_drawing_binding = self.canvas.mpl_connect('motion_notify_event', self.draw_peak_curve)
                self.canvas.draw()

    def release_add_peak(self, event):
        print('release')
        self.canvas.mpl_disconnect(self.peak_drawing_binding)
        fwhm = abs(self.add_peak_event.xdata - event.xdata)
        height = (event.ydata - self.add_peak_event.ydata)*self.Window.stackscale
        area = height * fwhm * 1.0645889
        if not ALLOW_NEGATIVE_PEAKS:
            area = abs(area)
        parameter = self.Window.parameters[np.abs(self.Window.parameters - self.add_peak_event.ydata).argmin()]
        peak = Peak(self.add_peak_event.xdata, area, fwhm, parameter, manual=True)
        print(peak)
        self.Window.peaks.append(peak)
        self.draw_fit_line.remove()
        self.Window.plot_peak_positions()
        self.Window.show_initial_parameters()
        self.Window.canvas.draw()

    def draw_peak_curve(self, event):
        x,y = event.xdata, event.ydata
        if event.inaxes:
            dx = abs(x - self.add_peak_event.xdata)
            dy = (y - self.add_peak_event.ydata)*self.Window.stackscale
            if not ALLOW_NEGATIVE_PEAKS:
                dy = abs(dy)
            xstart, xend = self.canvas.figure.axes[0].get_xlim()
            y = gaussian_width_fwhm(np.linspace(xstart, xend, 400), self.add_peak_event.xdata, dy, dx)
            self.draw_fit_line.set_ydata(y + self.add_peak_event.ydata*self.Window.stackscale)
            # plt.draw()
            self.canvas.draw()



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
    return area / (fwhm * np.sqrt(np.pi / (4*np.log(2)))) * np.exp(-4 * np.log(2) * (x - position) ** 2 / fwhm ** 2)

def gaussian_width_fwhm(x, position:float, height:float, fwhm:float):
    return height*np.exp(-4*np.log(2)*(x-position)**2 / fwhm**2)

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


class Peak():
    def __init__(self, position:float, area:float, fwhm:float, parameter:float, manual=False):
        self.position = position
        self.area = area
        self.fwhm = fwhm
        self.parameter = parameter
        self.manual = manual

    def __str__(self):
        return f'Peak ['f'pos: {self.position:.2f}, area: {self.area:.2f}, fwhm: {self.fwhm:.2f}, parameter: {self.parameter:.2f}]'


class MainWindow(QDialog):
    def __init__(self, datalist, parameters, parent=None):
        super(MainWindow, self).__init__(parent)
        self.parameters = parameters

        for i, data in enumerate(datalist):
            datalist[i] = np.array([data[0][np.isfinite(data[1])], data[1][np.isfinite(data[1])]])
        self.original_datalist = copy.deepcopy(datalist)
        self.datalist = copy.deepcopy(datalist)
        self.fitted_peak_parameters = []
        self.fit_errors = []
        self.peaks:List[Peak] = []
        self.fit_lines = []
        self.als_baselines = []
        self.initial_parameter_lines = []
        self.showInitialParameters = False
        self.resize(1440, 900)
        self.setWindowFlags(self.windowFlags() |
                            Qt.WindowSystemMenuHint |
                            Qt.WindowMinMaxButtonsHint)
        self.setWindowTitle('FitPeaks')

        # a figure instance to plot on
        self.figure = plt.figure(figsize=(10, 10))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = MyToolbar(self.canvas, self)
        button_press_id = self.figure.canvas.mpl_connect('button_press_event', self.onclick)
        # button_release_id = self.figure.canvas.mpl_connect('button_release_event', self.onrelease)


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

        # Show initial parameters
        self.showInitialParametersButton = QPushButton('Show fit pars')
        self.showInitialParametersButton.clicked.connect(self.initial_parameters_button_callback)

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
        self.plotButtonBox.addWidget(self.showInitialParametersButton)
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
        self.ax2 = self.ax.twinx()
        self.ax.callbacks.connect("ylim_changed", self.convert_ax_to_ax2)
        self.als_baselines = []
        self.als_baseline_lines = []
        self.fit_lines = []

        self.lines = plot_waterfall(self.datalist, self.parameters, self.ax, stackscale=self.stackscale, line_color='red')
        self.plot_peak_positions()
        self.show_initial_parameters()
        self.ax.set_xlabel('Wavenumber (cm$^{-1}$)')
        self.ax.set_ylabel('Absorption')
        self.ax2.set_ylabel('Parameter')
        self.figure.sca(self.ax)
        print(self.figure.gca())

        self.rectangle_selector = RectangleSelector(self.ax2, self.remove_peaks, drawtype='box', useblit=True, button=[3],
                                                    minspanx=5, minspany=5, spancoords='pixels', interactive=False)
        self.change_xlimit()
        self.canvas.draw()

    def convert_ax_to_ax2(self, ax):
        y1, y2 = ax.get_ylim()
        self.ax2.set_ylim(y1/self.stackscale, y2/self.stackscale)

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
        new_peaks = find_positive_peaks(self.datalist, threshold=float(self.thresholdValueBox.text()),
                                    min_dist=self.minDistSlider.value())

        for peak in self.peaks:
            if not peak.manual:
                try:
                    peak.scatter_point.remove()
                except (AttributeError, ValueError):
                    pass

        manual_peaks = [p for p in self.peaks if p.manual]
        self.peaks = [p for p in self.peaks if p.manual]
        for i,row in enumerate(new_peaks):
            parameter = self.parameters[i]
            for peak_position in row:
                for p in manual_peaks:
                    if parameter == p.parameter:
                        if abs(p.position - peak_position) < int(self.minDistValueBox.text()):
                            break
                else:
                    self.peaks.append(Peak(peak_position, INITIAL_PEAK_AREA, INITIAL_PEAK_FWHM, parameter))
        self.plot_peak_positions()

    def plot_peak_positions(self):

        xmin, xmax = self.get_xlimits()
        for peak in self.peaks:
            try:
                peak.scatter_point.remove()
            except (AttributeError, ValueError): pass
            if xmin <= peak.position <= xmax:
                color = 'k' if peak.manual else 'b'
                peak.scatter_point = self.ax.scatter(peak.position, peak.parameter*self.stackscale, c=color)
        self.canvas.draw()

    def fit_peaks(self):
        for peak in self.peaks:
            print(peak)
        self.progressBar.show()
        fit_data, lines, fit_errors = [], [], []
        als_baseline_calculated = len(self.als_baselines) > 0

        print('Fitting parameter: ', end='')
        for i, data in enumerate(self.datalist):
            print(f'{self.parameters[i]}, ', end='', flush=True)
            x, y = data
            if als_baseline_calculated:
                y = y - self.als_baselines[i][1]
            initial_parameters = [0, 0, 0]
            bounds_min = [-np.inf,-np.inf,-np.inf]
            bounds_max = [np.inf, np.inf, np.inf]
            peaks = [p for p in self.peaks if p.parameter == self.parameters[i]]
            for peak in peaks:
                try:
                    initial_parameters.extend([peak.fitted_position, peak.fitted_area, peak.fitted_fwhm])
                except AttributeError:
                    initial_parameters.extend([peak.position, peak.area, peak.fwhm])
                if ALLOW_NEGATIVE_PEAKS:
                    bounds_min.extend([peak.position-3, -np.inf, 0])
                else:
                    bounds_min.extend([peak.position-3, 0, 0])
                bounds_max.extend([peak.position+3, np.inf, np.inf])
            try:
                fitted_parameters, covariance = optimize.curve_fit(multiple_gaussian_with_baseline_correction, x, y,
                                                                   p0=initial_parameters, bounds=(bounds_min,bounds_max))
                for j,peak in enumerate(peaks):
                    peak.fitted_position = fitted_parameters[j*3+3]
                    peak.fitted_area = fitted_parameters[j*3+4]
                    peak.fitted_fwhm = fitted_parameters[j*3+5]
            except RuntimeError:
                print("\nERROR: Couldn't fit parameter: {}".format(self.parameters[i]))
                fitted_parameters, covariance = [0, 0, 0], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            fit_errors.append(np.sqrt(np.diag(covariance)))
            fit_data.append(fitted_parameters)
            self.progressBar.setValue(i)
        print('\n' + '*'*5 + '\t Fitting finished\t' + '*'*5)
        self.progressBar.hide()

        self.fitted_peak_parameters = fit_data
        self.fit_errors = fit_errors
        self.plot_fitted_data()
        self.show_initial_parameters()
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

    def initial_parameters_button_callback(self):
        if self.showInitialParameters == True:
            self.showInitialParametersButton.setText('Show fit pars')
            self.showInitialParameters = False
        else:
            self.showInitialParametersButton.setText('Hide fit pars')
            self.showInitialParameters = True
        self.show_initial_parameters()

    def show_initial_parameters(self):
        for line in self.initial_parameter_lines:
            try: line.remove()
            except ValueError: pass
        if self.showInitialParameters == True:
            self.initial_parameter_lines = []
            xmin, xmax = self.get_xlimits()
            x = np.linspace(xmin, xmax, 500)
            for i,par in enumerate(self.parameters):
                if len(self.fitted_peak_parameters) > 0:
                    initial_parameters = list(self.fitted_peak_parameters[i][:3])
                else:
                    initial_parameters = [0, 0, 0]
                peaks = [p for p in self.peaks if p.parameter == par]
                for peak in peaks:
                    try:
                        initial_parameters.extend([peak.fitted_position, peak.fitted_area, peak.fitted_fwhm])
                    except AttributeError:
                        initial_parameters.extend([peak.position, peak.area, peak.fwhm])
                y = multiple_gaussian_with_baseline_correction(x, *initial_parameters)
                line, = self.ax.plot(x, y+self.stackscale*par, '--', color='k', linewidth=1)
                self.initial_parameter_lines.append(line)

        self.canvas.draw()

    def draw_fit_curve(self, event):
        x,y = event.xdata, event.ydata
        if event.inaxes:
            dx = abs(x - self.left_click_event.xdata)
            dy = abs(y - self.left_click_event.ydata)
            xstart, xend = self.get_xlimits()
            y = gaussian_width_fwhm(np.linspace(xstart, xend, 400), self.left_click_event.xdata, dy, dx)
            self.draw_fit_line.set_ydata(y + self.left_click_event.ydata)
            # plt.draw()
            self.canvas.draw()


    def onclick(self, event):
        print(self.figure.gca())
        if event.button == 1 and event.dblclick:
            parameter = self.parameters[np.abs(self.parameters - event.ydata).argmin()]
            peak = Peak(event.xdata, INITIAL_PEAK_AREA, INITIAL_PEAK_FWHM, parameter, manual=True)
            print(peak)
            self.peaks.append(peak)
            self.plot_peak_positions()
            self.show_initial_parameters()



    def remove_peaks(self, eclick, erelease):
        '''
        Removes peak positions in selected area
        :param eclick: mouse press event
        :param erelease: mouse release event
        '''
        x = eclick.xdata, erelease.xdata
        y = eclick.ydata, erelease.ydata
        peaks = []
        for peak in self.peaks:
            if (min(x) < peak.position < max(x)) and (min(y) < peak.parameter < max(y)):
                peak.scatter_point.remove()
            else:
                peaks.append(peak)
        self.peaks = peaks
        self.plot_peak_positions()
        self.show_initial_parameters()

    def remove_all_peaks(self):
        for peak in self.peaks:
            try:
                peak.scatter_point.remove()
            except (AttributeError, ValueError):
                pass
        self.peaks = []
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


class SelectFilesWindow(QDialog):
    def __init__(self, parent=None):
        super(SelectFilesWindow, self).__init__(parent)
        self.resize(600, 600)
        self.setWindowTitle('Select peaks for fitting')
        self.layout = QHBoxLayout()
        self.left_layout = QVBoxLayout()

        self.filenameDecimalCharacter = 'T'
        self.filenameDelimiter = '_'

        # self.spectraFullFileNames = []
        self.spectraFileNamesTable = QTableWidget()
        self.spectraFileNamesTable.setColumnCount(2)
        self.spectraFileNamesTable.setHorizontalHeaderLabels(['Filename','Parameter'])

        self.selectFilesButton = QPushButton('Select files')
        self.selectFilesButton.clicked.connect(self.select_spectra_names_dialog)
        self.left_layout.addWidget(self.selectFilesButton)

        self.filenameDecimalCharacterLabel = QLabel(self)
        self.filenameDecimalCharacterLabel.setText('Decimal:')
        self.filenameDecimalCharacter = QLineEdit(self)
        self.filenameDecimalCharacter.setText('T')
        self.filenameDecimalCharacter.setFixedWidth(50)
        self.filenameDecimalCharacter.editingFinished.connect(self.update_parameters)
        self.left_layout.addWidget(self.filenameDecimalCharacterLabel)
        self.left_layout.addWidget(self.filenameDecimalCharacter)

        self.filenameDelimiterLabel = QLabel(self)
        self.filenameDelimiterLabel.setText('Delimiter:')
        self.filenameDelimiter = QLineEdit(self)
        self.filenameDelimiter.setText('_')
        self.filenameDelimiter.setFixedWidth(50)
        self.filenameDelimiter.editingFinished.connect(self.update_parameters)
        self.left_layout.addWidget(self.filenameDelimiterLabel)
        self.left_layout.addWidget(self.filenameDelimiter)

        self.left_layout.addStretch(1)
        self.selectFilesButton = QPushButton('FIT')
        self.selectFilesButton.clicked.connect(self.send_to_fit)
        self.left_layout.addWidget(self.selectFilesButton)

        self.layout.addLayout(self.left_layout)
        self.layout.addWidget(self.spectraFileNamesTable)
        self.setLayout(self.layout)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        self.fill_table(files)

    def select_spectra_names_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "Open files to be fitted", "", "*.dat files (*.dat);;All Files (*)", options=options)
        if files:
            self.fill_table(files)

    def fill_table(self, files):
        self.spectraFileNamesTable.clearContents()
        self.spectraFileNamesTable.setRowCount(len(files))
        for i, filename in enumerate([os.path.basename(f) for f in files]):
            table_item = QTableWidgetItem(filename)
            table_item.setData(Qt.UserRole, files[i])
            self.spectraFileNamesTable.setItem(i, 0, table_item)
        self.update_parameters()
        header = self.spectraFileNamesTable.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)

    def update_parameters(self):
        errors = False
        for i in range(self.spectraFileNamesTable.rowCount()):
            filename = self.spectraFileNamesTable.item(i,0).text()
            try:
                par = find_num_from_name(filename, self.filenameDecimalCharacter.text(), self.filenameDelimiter.text())
            except (AttributeError, ValueError):
                par = 'Not found'
                errors = True
            self.spectraFileNamesTable.setItem(i, 1, QTableWidgetItem(str(par)))
            if par == 'Not found':
                self.spectraFileNamesTable.item(i, 1).setBackground(QColor(255, 0, 0))
        if errors:
            self.selectFilesButton.setEnabled(False)
        else:
            self.selectFilesButton.setEnabled(True)


    def send_to_fit(self):
        name_list = []
        parameters = []
        for i in range(self.spectraFileNamesTable.rowCount()):
            name_list.append(self.spectraFileNamesTable.item(i,0).data(Qt.UserRole))
        datalist = [np.loadtxt(filename, unpack=True) for filename in name_list]
        for name in name_list:
            parameters.append(find_num_from_name(name, self.filenameDecimalCharacter.text(), self.filenameDelimiter.text()))
        main = MainWindow(datalist, parameters)
        main.show()


def run(datalist, parameters):
    app = 0  # Fix for spyder crashing on consequent runs
    app = QApplication(sys.argv)
    main = MainWindow(datalist, parameters)
    main.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    app = 0  # Fix for spyder crashing on consequent runs
    app = QApplication(sys.argv)

    # This part needed for taskbar icon, see here: https://stackoverflow.com/questions/1551605/how-to-set-applications-taskbar-icon-in-windows-7
    import ctypes
    myappid = u'mycompany.myproduct.subproduct.version'  # arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    app.setWindowIcon(QIcon('images/if_trends_1054952-512.png'))

    select_files = SelectFilesWindow()
    select_files.show()
    sys.exit(app.exec_())


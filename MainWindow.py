import copy
import os

import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QDialog, QPushButton, QSlider, QLineEdit, QLabel, QHBoxLayout, QVBoxLayout, \
    QFileDialog, QProgressBar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import RectangleSelector
from scipy import optimize

from CustomToolbar import CustomToolbar
from FitResultWindow import FitResultWindow
from Peak import Peak
from fitpeaks import INITIAL_PEAK_AREA, INITIAL_PEAK_FWHM, ALLOW_NEGATIVE_PEAKS
from helper_functions import *


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
        self.peaks: List[Peak] = []
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
        self.toolbar = CustomToolbar(self.canvas, self)
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
        self.ALSLambdaValueBox.editingFinished.connect(self.ALSLambda_valuebox_changed)
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
        self.ALSPositiveWeightValueBox.editingFinished.connect(self.ALSPositiveWeight_valuebox_changed)
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

    def ALSLambda_valuebox_changed(self):
        value = float(self.ALSLambdaValueBox.text()) * 10
        self.ALSLambdaSlider.setValue(int(value))
        self.calculate_als_baseline()

    def ALSPositiveWeight_valuebox_changed(self):
        value = float(self.ALSPositiveWeightValueBox.text()) * 100000
        self.ALSPositiveWeightSlider.setValue(int(value))
        self.calculate_als_baseline()

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
                line = self.ax.plot(als_baselines[i][0], als_baselines[i][1] + self.parameters[i] * self.stackscale,
                                    'g')
                lines.extend(line)
                self.als_baseline_lines = lines
        self.als_baselines = als_baselines
        self.canvas.draw()

    def subtract_als_from_datalist(self):
        for i, data in enumerate(self.datalist):
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

        self.lines = plot_waterfall(self.datalist, self.parameters, self.ax, stackscale=self.stackscale,
                                    line_color='red')
        self.plot_peak_positions()
        self.show_initial_parameters()
        self.ax.set_xlabel('Wavenumber (cm$^{-1}$)')
        self.ax.set_ylabel('Absorption')
        self.ax2.set_ylabel('Parameter')
        self.figure.sca(self.ax)

        self.rectangle_selector = RectangleSelector(self.ax2, self.remove_peaks, drawtype='box', useblit=True,
                                                    button=[3],
                                                    minspanx=5, minspany=5, spancoords='pixels', interactive=False)
        self.change_xlimit()
        self.canvas.draw()

    def convert_ax_to_ax2(self, ax):
        y1, y2 = ax.get_ylim()
        self.ax2.set_ylim(y1 / self.stackscale, y2 / self.stackscale)

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
            ymin = np.min([np.min(x[1]) + self.parameters[i] * self.stackscale for i, x in enumerate(self.datalist)])
            ymax = np.max([np.max(x[1]) + self.parameters[i] * self.stackscale for i, x in enumerate(self.datalist)])
        else:
            ymax = np.max([np.max(x[1]) + self.parameters[i] * self.stackscale for i, x in enumerate(self.datalist)])
            ymin = np.min([np.min(x[1]) + self.parameters[i] * self.stackscale for i, x in enumerate(self.datalist)])
        delta = abs(ymax - ymin) * 0.02
        self.ax.set_ylim((ymin - delta, ymax + delta))

    def stackscale_valuebox_changed(self):
        self.stackscaleSlider.setValue(int(self.stackscaleValueBox.text()))
        self.change_stackscale()

    def stackscale_slider_changed(self):
        self.stackscaleValueBox.setText(str(self.stackscaleSlider.value()))
        self.change_stackscale()

    def change_stackscale(self):
        self.stackscale = self.stackscaleSlider.value()
        for i, line in enumerate(self.lines):
            line.set_ydata(self.datalist[i][1] + self.stackscale * self.parameters[i])
        self.plot_peak_positions()
        if len(self.fit_lines) > 0:
            self.plot_fitted_data()
        for i, line in enumerate(self.als_baseline_lines):
            line.set_ydata(self.als_baselines[i][1] + self.stackscale * self.parameters[i])
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
        for i, row in enumerate(new_peaks):
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
            except (AttributeError, ValueError):
                pass
            if xmin <= peak.position <= xmax:
                color = 'k' if peak.manual else 'b'
                peak.scatter_point = self.ax.scatter(peak.position, peak.parameter * self.stackscale, c=color)
        self.canvas.draw()

    def fit_peaks(self):
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
            bounds_min = [-np.inf, -np.inf, -np.inf]
            bounds_max = [np.inf, np.inf, np.inf]
            peaks = [p for p in self.peaks if p.parameter == self.parameters[i]]
            for peak in peaks:
                try:
                    initial_parameters.extend([peak.fitted_position, peak.fitted_area, peak.fitted_fwhm])
                except AttributeError:
                    initial_parameters.extend([peak.position, peak.area, peak.fwhm])
                if ALLOW_NEGATIVE_PEAKS:
                    bounds_min.extend([peak.position - 3, -np.inf, 0])
                else:
                    bounds_min.extend([peak.position - 3, 0, 0])
                bounds_max.extend([peak.position + 3, np.inf, np.inf])
            try:
                fitted_parameters, covariance = optimize.curve_fit(multiple_gaussian_with_baseline_correction, x, y,
                                                                   p0=initial_parameters,
                                                                   bounds=(bounds_min, bounds_max))
                for j, peak in enumerate(peaks):
                    peak.fitted_position = fitted_parameters[j * 3 + 3]
                    peak.fitted_area = fitted_parameters[j * 3 + 4]
                    peak.fitted_fwhm = fitted_parameters[j * 3 + 5]
            except RuntimeError:
                print("\nERROR: Couldn't fit parameter: {}".format(self.parameters[i]))
                fitted_parameters, covariance = [0, 0, 0], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            fit_errors.append(np.sqrt(np.diag(covariance)))
            fit_data.append(fitted_parameters)
            self.progressBar.setValue(i)
        print('\n' + '*' * 5 + '\t Fitting finished\t' + '*' * 5)
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
            for i, data in enumerate(self.datalist):
                als_baseline = self.als_baselines[i][1] if als_baseline_calculated else 0
                if lines_exist_already:
                    fit_line = multiple_gaussian_with_baseline_correction(x, *self.fitted_peak_parameters[i])
                    self.fit_lines[i].set_ydata(fit_line + self.parameters[i] * self.stackscale + als_baseline)
                else:
                    line = self.ax.plot(x,
                                        multiple_gaussian_with_baseline_correction(x, *self.fitted_peak_parameters[i]) +
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
        self.FitResultPlot = FitResultWindow(self.fitted_peak_parameters, self.fit_errors, self.parameters,
                                             self.ax.get_xlim())
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
            try:
                line.remove()
            except ValueError:
                pass
        if self.showInitialParameters == True:
            self.initial_parameter_lines = []
            xmin, xmax = self.get_xlimits()
            x = np.linspace(xmin, xmax, 500)
            for i, par in enumerate(self.parameters):
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
                line, = self.ax.plot(x, y + self.stackscale * par, '--', color='k', linewidth=1)
                self.initial_parameter_lines.append(line)

        self.canvas.draw()

    def draw_fit_curve(self, event):
        x, y = event.xdata, event.ydata
        if event.inaxes:
            dx = abs(x - self.left_click_event.xdata)
            dy = abs(y - self.left_click_event.ydata)
            xstart, xend = self.get_xlimits()
            y = gaussian_width_fwhm(np.linspace(xstart, xend, 400), self.left_click_event.xdata, dy, dx)
            self.draw_fit_line.set_ydata(y + self.left_click_event.ydata)
            # plt.draw()
            self.canvas.draw()

    def onclick(self, event):
        if event.button == 1 and event.dblclick:
            parameter = self.parameters[np.abs(self.parameters - event.ydata).argmin()]
            peak = Peak(event.xdata, INITIAL_PEAK_AREA, INITIAL_PEAK_FWHM, parameter, manual=True)
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

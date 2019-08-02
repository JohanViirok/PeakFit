import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QDialog, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class FitResultWindow(QDialog):
    def __init__(self, fitted_parameters, fit_errors, parameters, xlimits):
        super(FitResultWindow, self).__init__()
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

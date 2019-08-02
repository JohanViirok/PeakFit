import numpy as np
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from Peak import Peak
from fitpeaks import ALLOW_NEGATIVE_PEAKS
from helper_functions import gaussian_width_fwhm


class CustomToolbar(NavigationToolbar):
    def __init__(self, figure_canvas, parent=None):
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
        NavigationToolbar.__init__(self, figure_canvas, parent=None)
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
        super(CustomToolbar, self)._update_buttons_checked()
        self._actions['add_peak_tool'].setChecked(self._active == 'ADD')

    def press_add_peak(self, event):
        if event.button == 1:
            if event.inaxes is not None:
                self.add_peak_event = event
                x = np.linspace(*self.canvas.figure.axes[0].get_xlim(), 400)
                self.draw_fit_line, = self.canvas.figure.axes[0].plot(x, gaussian_width_fwhm(x, event.xdata, 0,
                                                                                             0) + event.ydata * self.Window.stackscale,
                                                                      c='k')
                self.peak_drawing_binding = self.canvas.mpl_connect('motion_notify_event', self.draw_peak_curve)
                self.canvas.draw()

    def release_add_peak(self, event):
        self.canvas.mpl_disconnect(self.peak_drawing_binding)
        fwhm = abs(self.add_peak_event.xdata - event.xdata)
        height = (event.ydata - self.add_peak_event.ydata) * self.Window.stackscale
        area = height * fwhm * 1.0645889
        if not ALLOW_NEGATIVE_PEAKS:
            area = abs(area)
        parameter = self.Window.parameters[np.abs(self.Window.parameters - self.add_peak_event.ydata).argmin()]
        peak = Peak(self.add_peak_event.xdata, area, fwhm, parameter, manual=True)
        self.Window.peaks.append(peak)
        self.draw_fit_line.remove()
        self.Window.plot_peak_positions()
        self.Window.show_initial_parameters()
        self.Window.canvas.draw()

    def draw_peak_curve(self, event):
        x, y = event.xdata, event.ydata
        if event.inaxes:
            dx = abs(x - self.add_peak_event.xdata)
            dy = (y - self.add_peak_event.ydata) * self.Window.stackscale
            if not ALLOW_NEGATIVE_PEAKS:
                dy = abs(dy)
            xstart, xend = self.canvas.figure.axes[0].get_xlim()
            y = gaussian_width_fwhm(np.linspace(xstart, xend, 400), self.add_peak_event.xdata, dy, dx)
            self.draw_fit_line.set_ydata(y + self.add_peak_event.ydata * self.Window.stackscale)
            # plt.draw()
            self.canvas.draw()

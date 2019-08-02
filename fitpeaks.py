import os
import sys

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication

import MainWindow
# from MainWindow import MainWindow
from SelectFilesWindow import SelectFilesWindow

INITIAL_PEAK_AREA = 10
INITIAL_PEAK_FWHM = 1
ALLOW_NEGATIVE_PEAKS = True


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
    image_path = 'images/icon.png'
    if os.path.exists(image_path):
        import ctypes

        myappid = u'kbfi.peakfit.1.0'  # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        app.setWindowIcon(QIcon(image_path))

    select_files = SelectFilesWindow()
    select_files.show()
    sys.exit(app.exec_())

import os

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QDialog, QPushButton, QLineEdit, QLabel, QHBoxLayout, QVBoxLayout, \
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView

import MainWindow
from helper_functions import find_num_from_name


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
        self.spectraFileNamesTable.setHorizontalHeaderLabels(['Filename', 'Parameter'])

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
        files, _ = QFileDialog.getOpenFileNames(self, "Open files to be fitted", "",
                                                "*.dat files (*.dat);;All Files (*)", options=options)
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
            filename = self.spectraFileNamesTable.item(i, 0).text()
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
            name_list.append(self.spectraFileNamesTable.item(i, 0).data(Qt.UserRole))
        datalist = [np.loadtxt(filename, unpack=True) for filename in name_list]
        for name in name_list:
            parameters.append(
                find_num_from_name(name, self.filenameDecimalCharacter.text(), self.filenameDelimiter.text()))
        main = MainWindow.MainWindow(datalist, parameters)
        main.show()

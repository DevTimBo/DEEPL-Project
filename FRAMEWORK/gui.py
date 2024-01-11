from PyQt5 import QtWidgets, uic
import sys
import cv2 as cv
import FRAMEWORK.LRP as LRP

from PyQt5.QtWidgets import (qApp, QFileDialog,
                             QListWidgetItem)


class Ui(QtWidgets.QDialog):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('TestUI.ui', self)

        self.images = []
        self.pushButtonL.clicked.connect(self.open_file_dialog)
        self.pushButtonK.clicked.connect(self.analyze)
        self.pushButtonK.setEnabled(False)
        self.show()

    def analyze(self):
        if self.comboBox.currentText() == "LRP":
            LRP.lrp_simple(self.images)
        elif self.comboBox.currentText() == "LIME":
            pass

    def closeEvent(self, event):
        self.hide()
        qApp.quit()

    def open_file_dialog(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("Images (*.png *.jpg *.bmp)")
        dialog.setViewMode(QFileDialog.ViewMode.List)
        if dialog.exec():
            filename = dialog.selectedFiles()
            self.load_picture(filename[0])
            self.pushButtonK.setEnabled(True)

    def add_image_to_list(self, filename):
        item = QListWidgetItem(filename)
        self.image_list_widget.addItem(item)

    def load_picture(self, filename):
        self.add_image_to_list(filename)
        image = cv.imread(filename)
        self.images.append(image)


app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()

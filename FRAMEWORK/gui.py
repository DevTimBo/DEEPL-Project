from PyQt5 import QtWidgets, uic
import sys
import cv2 as cv
import FRAMEWORK.LRP as LRP
import FRAMEWORK.LIME as LIME
import FRAMEWORK.IMAGE_EDIT as IMAGE_EDIT
import FRAMEWORK.PLOTTING as PLOTTING
import FRAMEWORK.GRAD_CAM as GRAD_CAM
import datei_laden
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from PyQt5.QtWidgets import (QApplication, QDialog,QTextEdit,
                             QPushButton,QVBoxLayout,QLabel,QWidget,
                             QHBoxLayout,qApp,QSpinBox, QDoubleSpinBox,
                             QGridLayout, QFormLayout,QLineEdit,
                             QDateEdit, QComboBox,
                             QMainWindow,
    QWidget,
    QFileDialog,
    QGridLayout,
    QPushButton,
    QLabel,
    QAction,QListWidgetItem,

    QListWidget)


class Ui(QtWidgets.QDialog):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('TestUI.ui', self)

        self.single_image = ""
        self.many_images = []
        self.video = ""
        self.button_load_single_image.clicked.connect(self.file_dialog_single)
        self.button_load_many_images.clicked.connect(self.file_dialog_many)
        self.button_load_video.clicked.connect(self.file_dialog_video)
        self.button_analyze.clicked.connect(self.analyze)
        self.show()


    def load_picture(self, filepath):
        self.single_image = filepath
        self.single_image_label.setPixmap(datei_laden.datei_to_pixmap(filepath))

    def file_dialog_single(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("Images (*.png *.jpg *.bmp)")
        dialog.setViewMode(QFileDialog.ViewMode.List)
        if dialog.exec():
            filename = dialog.selectedFiles()
            self.load_picture(filename[0])

    def add_images_to_list(self, filenames):
        for filename in filenames:
            item = QListWidgetItem(filename)
            self.image_list_widget.addItem(item)

    def file_dialog_many(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("Images (*.png *.jpg *.bmp)")
        dialog.setViewMode(QFileDialog.ViewMode.List)
        if dialog.exec():
            filenames = dialog.selectedFiles()
            self.add_images_to_list(filenames)

    def load_video(self, filepath):
        self.video = filepath
        self.video_text.setText(filepath)

    def file_dialog_video(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("Video Files (*.avi *.mp4 *.flv)")
        dialog.setViewMode(QFileDialog.ViewMode.List)
        if dialog.exec():
            filename = dialog.selectedFiles()
            self.load_video(filename[0])
    def analyze(self):

        if self.model.currentText() == "VGG16":
            import keras.applications.vgg16 as vgg16
            # Keras Model
            model = vgg16.VGG16(weights="imagenet")
            preprocess = vgg16.preprocess_input
            decode_predictions = vgg16.decode_predictions
            last_conv_layer = "block5_conv3"
        elif self.model.currentText() == "VGG19":
            pass
        elif self.model.currentText() == "ResNet":
            pass

        if self.analyze_mode.currentText() == "Single Image":
            if self.single_image != "":
                self.single_image_analyzer(model, preprocess, decode_predictions, last_conv_layer)
        elif self.analyze_mode.currentText() == "Many Images":
            pass
        elif self.analyze_mode.currentText() == "Video":
            pass

    def single_image_analyzer(self, model, preprocess, decode_predictions, last_conv_layer):
        image_list = []
        title_list = []
        cmap_list = []
        image = cv.imread(self.single_image)
        if self.model.currentText() == "VGG16":
            size = (224, 224)
            resized_image = cv.resize(image, size)
        elif self.model.currentText() == "VGG19":
            pass
        elif self.model.currentText() == "ResNet":
            pass

        image_list.append(resized_image)
        title_list.append("Original")
        cmap_list.append("viridis")
        if self.lrp_checkbox.isChecked():
            print("LRP")
            rule = self.lrp_rule_box.currentText()
            lrp_image = LRP.analyze_image_lrp(resized_image, model, preprocess, rule)
            title_list.append(f"LRP: {rule}")
            cmap_list.append('viridis')
            image_list.append(lrp_image)
        if self.gradcam_checkbox.isChecked():
            pass
            # print("GRAD_CAM")
            # gradcam_image = GRAD_CAM.make_grad_cam(model, self.single_image, size, preprocess,
            #                                        decode_predictions, last_conv_layer)
            # print(gradcam_image.shape)
            # image_list.append(gradcam_image)
            # title_list.append(f"GRAD CAM")
            # cmap_list.append('viridis')
        if self.lime_checkbox.isChecked():
            samples = self.lime_samples_box.value()
            lime_image = LIME.get_lime_explanation(resized_image, model, samples)
            title_list.append(f"LIME {samples}")
            cmap_list.append('viridis')
            image_list.append(lime_image)
        if self.overlap_box.isChecked():
            print("OVERLAP")
            if len(image_list) > 1:
                overlap_images = image_list[1:]
                avg_image = IMAGE_EDIT.overlap_images(overlap_images)
                title_list.append('AVG IMAGE')
                image_list.append(avg_image)

                cmap_list.append('viridis')

        PLOTTING.plot_n_images(image_list, title_list, cmap_list, figsize=(20, 5))
    def closeEvent(self, event):
        self.hide()
        qApp.quit()







app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()

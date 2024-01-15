from PyQt5 import QtWidgets, uic
import sys
import cv2 as cv
import LRP.LRP as LRP
import LIME.LIME as LIME
from utils import datei_laden, IMAGE_EDIT, PLOTTING
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from PyQt5.QtWidgets import (qApp, QFileDialog,
                             QListWidgetItem)


class Ui(QtWidgets.QDialog):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('GUI.ui', self)

        # Current Keras Model
        self.keras_model = -1
        self.keras_preprocess = -1
        self.keras_decode = -1
        self.last_conv_layer = -1

        # Loaded Files
        self.single_image = ""
        self.many_images = []
        self.video = ""

        # Buttons
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
        # TODO New THread
        if self.model.currentText() == "VGG16":
            import keras.applications.vgg16 as vgg16
            # Keras Model
            self.keras_model = vgg16.VGG16(weights="imagenet")
            self.keras_preprocess = vgg16.preprocess_input
            self.keras_decode = vgg16.decode_predictions
            self.last_conv_layer = "block5_conv3"
        elif self.model.currentText() == "VGG19":
            pass
        elif self.model.currentText() == "ResNet":
            pass

        if self.analyze_mode.currentText() == "Single Image":
            if self.single_image != "":
                self.single_image_analyzer()
        elif self.analyze_mode.currentText() == "Many Images":
            if self.many_images != []:
                self.many_images_analyzer()
        elif self.analyze_mode.currentText() == "Video":
            pass

    def single_image_analyzer(self):
        image_list = []
        title_list = []
        cmap_list = []
        image = cv.imread(self.single_image, cv.IMREAD_COLOR)
        if self.model.currentText() == "VGG16":
            size = (224, 224)
            resized_image = cv.resize(image, size)
        elif self.model.currentText() == "VGG19":
            pass
        elif self.model.currentText() == "ResNet":
            pass
        resized_image = convert_to_uint8(resized_image)
        image_list.append(resized_image)
        title_list.append(f"Original")
        cmap_list.append("viridis")

        if self.noise_checkbox.isChecked():
            noise_level = self.noiselevel_box.value()
            resized_image = IMAGE_EDIT.add_noise(resized_image, noise_level)

            image_list.append(resized_image)
            title_list.append("Noise")
            cmap_list.append("viridis")
            cv.imwrite("data/noise_image.png", resized_image)

        if self.lrp_checkbox.isChecked():
            rule = self.lrp_rule_box.currentText()
            lrp_image = self.lrp_analyze(resized_image, rule)

            title_list.append(f"LRP: {rule}")
            cmap_list.append('viridis')
            image_list.append(lrp_image)

        if self.gradcam_checkbox.isChecked():
            grad_cam_image = self.grad_cam_analyze()

            title_list.append(f"GRAD CAM")
            cmap_list.append('viridis')
            image_list.append(grad_cam_image)

        if self.lime_checkbox.isChecked():
            samples = self.lime_samples_box.value()
            features = self.lime_features_box.value()
            lime_image = self.lime_analyzer(image, samples, features)

            title_list.append(f"LIME {samples}, Top Features: {features}")
            cmap_list.append('viridis')
            image_list.append(lime_image)

        if self.overlap_box.isChecked():
            overlap_image = self.overlap_images(image_list)
            print("Overlap Image")
            overlap_image = convert_to_uint8(overlap_image)
            title_list.append('Overlap IMAGE')
            image_list.append(overlap_image)
            cmap_list.append('viridis')

        print("Plotting")
        PLOTTING.plot_n_images(image_list, title_list, cmap_list, figsize=(20, 5))
        
    def many_images_analyzer(self):
        images_info = []

        for i in range(len(self.many_images)):
            print(f"Processing image {i + 1}/{len(self.many_images)}")

            image = cv.imread(self.many_images[i], cv.IMREAD_COLOR)

            if self.model.currentText() == "VGG16":
                size = (224, 224)
                resized_image = cv.resize(image, size)
            elif self.model.currentText() == "VGG19":
                pass
            elif self.model.currentText() == "ResNet":
                pass

            resized_image = convert_to_uint8(resized_image)

            # Original image information
            info_original = {
                'image': resized_image,
                'title': f"Original" + i,
                'cmap': "viridis"
            }
            images_info.append(info_original)

            # Add noise if the checkbox is checked
            if self.noise_checkbox.isChecked():
                noise_level = self.noiselevel_box.value()
                resized_image = IMAGE_EDIT.add_noise(resized_image, noise_level)

                # Noisy image information
                info_noisy = {
                    'image': resized_image,
                    'title': "Noise" + i,
                    'cmap': "viridis"
                }
                images_info.append(info_noisy)
                cv.imwrite("data/noise_image{i}.png", resized_image)

            if self.lrp_checkbox.isChecked():
                rule = self.lrp_rule_box.currentText()
                lrp_image = self.lrp_analyze(resized_image, rule)

                # lrp image information
                info_lrp = {
                    'image': lrp_image,
                    'title': "LRP: " + rule + i,
                    'cmap': "viridis"
                }
                images_info.append(info_lrp)
                cv.imwrite("data/lrp_image{i}.png", lrp_image)

            if self.gradcam_checkbox.isChecked():
                grad_cam_image = self.grad_cam_analyze()

                # gradcam image information
                info_grad = {
                    'image': grad_cam_image,
                    'title': "GRADCAM" + i,
                    'cmap': "viridis"
                }
                images_info.append(info_grad)
                cv.imwrite("data/gc_image{i}.png", grad_cam_image)

            if self.lime_checkbox.isChecked():
                samples = self.lime_samples_box.value()
                features = self.lime_features_box.value()
                lime_image = self.lime_analyzer(image, samples, features)
                #lime_image = self.lime_heatmap(image, samples)

                # lime image information
                info_lime = {
                    'image': lime_image,
                    'title': "LIME: " + samples + features + i,
                    'cmap': "viridis"
                }
                images_info.append(info_lime)
                cv.imwrite("data/lime_image{i}.png", lime_image)

            if self.overlap_box.isChecked():
                overlap_image = self.overlap_images(images_info['image'])
                print("Overlapped Image" + i)
                overlap_image = convert_to_uint8(overlap_image)
                # overlap image information
                info_overlap = {
                    'image': overlap_image,
                    'title': "Overlap: " + i,
                    'cmap': "viridis"
                }
                images_info.append(info_overlap)
                cv.imwrite("data/overlap_image{i}.png", overlap_image)

        print("Plotting")
        PLOTTING.plot_images_info(images_info, figsize=(20, 5))

    def lrp_analyze(self, image, rule):
        print("LRP")

        lrp_image = LRP.analyze_image_lrp(image, self.keras_model, self.keras_preprocess, rule)
        lrp_image = convert_to_uint8(lrp_image)
        zeros_array = np.zeros_like(lrp_image)
        lrp_image = cv.merge([zeros_array, zeros_array, lrp_image])  # LRP nur im roten Kanal
        return lrp_image

    def grad_cam_analyze(self):
        # TODO Pickle Model Übergeben, Last Conv Layer
        print("GRAD_CAM")
        import subprocess
        # Specify the path to the TensorFlow script
        tensorflow_script_path = "CAM/framework_grad_cam.py"
        # Specify the model_name and filepath as arguments
        model_name = self.model.currentText()
        if self.noise_checkbox.isChecked():
            filepath = 'data/noise_image.png'
        else:
            filepath = self.single_image
        # Run the TensorFlow script as a subprocess with arguments
        subprocess.run(["python", tensorflow_script_path, model_name, filepath])
        grad_cam_image = cv.imread("data/grad_cam.jpg")
        grad_cam_image = convert_to_uint8(grad_cam_image)
        return grad_cam_image

    def lime_analyzer(self, image, samples, features):
        print("LIME_ANALYZER")
        lime_image = LIME.get_lime_explanation(image, self.keras_model, samples, features)
        lime_image = convert_to_uint8(lime_image)
        return lime_image
    # TODO heatmap (?) 
    def lime_heatmap(self, image, samples):
        print("LIME_HEATMAP")
        heatmap = LIME.get_lime_heat_map(image, self.keras_model, samples)
        heatmap = convert_to_uint8(heatmap)
        return heatmap

    def overlap_images(self, image_list):
        print("OVERLAP")
        if len(image_list) > 1:
            overlap_images = image_list[1:]
            for image in overlap_images:
                print(image.shape)
            overlap_image = IMAGE_EDIT.overlap_images(overlap_images)

        return overlap_image


    def closeEvent(self, event):
        self.hide()
        qApp.quit()


def convert_to_uint8(image):
    if image.dtype == np.float32 or image.dtype == np.float64:
        # Scale float values to the range [0, 255] and convert to uint8
        return (image * 255).clip(0, 255).astype(np.uint8)
    elif image.dtype == np.uint8:
        # Image is already uint8, no need to convert
        return image
    else:
        # Handle other data types or raise an error if needed
        raise ValueError("Unsupported data type. Supported types are float32, float64, and uint8.")


app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()

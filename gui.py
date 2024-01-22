from PyQt5 import QtWidgets, uic
import sys
import cv2 as cv
import LRP.LRP as LRP
import LIME.LIME as LIME
from MonteCarloDropout import mcd
from utils import datei_laden, IMAGE_EDIT, PLOTTING
import numpy as np
import tensorflow as tf
import threading
from VideoPlayer import VideoPlayer
from PyQt5.QtWidgets import QVBoxLayout

tf.compat.v1.disable_eager_execution()
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (qApp, QFileDialog,
                             QListWidgetItem)


class Ui(QtWidgets.QDialog):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('GUI.ui', self)

        # open qss file
        File = open("ui.qss", 'r')

        with File:
            qss = File.read()
            app.setStyleSheet(qss)
        # Current Keras Model
        self.keras_model = -1
        self.keras_preprocess = -1
        self.keras_decode = -1
        self.last_conv_layer = -1

        # Loaded Files
        self.single_image_path = ""
        self.many_images_paths = []
        self.video_path = ""

        # Initialize the video player
        self.video_player = VideoPlayer()
        self.video_layout.addWidget(self.video_player)
        
        # Buttons
        self.button_load_single_image.clicked.connect(self.file_dialog_single)
        self.button_load_many_images.clicked.connect(self.file_dialog_many)
        self.button_load_video.clicked.connect(self.file_dialog_video)
        self.button_analyze.clicked.connect(self.analyze)

        # Connect the QListWidget
        self.image_list_widget.itemClicked.connect(self.show_selected_image)

        self.show()

    def load_picture(self, filepath):
        self.single_image_path = filepath
        self.single_image_label.setPixmap(datei_laden.datei_to_pixmap(filepath))

    def show_selected_image(self, item):
        # Display the selected image in the QLabel
        filepath = item.text()
        pixmap = datei_laden.datei_to_pixmap(filepath)
        self.many_images_label.setPixmap(pixmap)

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
            self.many_images_paths.append(filename)

    def file_dialog_many(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("Images (*.png *.jpg *.bmp)")
        dialog.setViewMode(QFileDialog.ViewMode.List)
        if dialog.exec():
            filenames = dialog.selectedFiles()
            self.add_images_to_list(filenames)

    def load_video(self, filepath):
        self.video_path = filepath
        self.video_text.setText(filepath)
        self.video_player.load_video(filepath)

    def file_dialog_video(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("Video Files (*.avi *.mp4 *.flv)")
        dialog.setViewMode(QFileDialog.ViewMode.List)
        if dialog.exec():
            filename = dialog.selectedFiles()
            self.load_video(filename[0])

    # Starting New Thread so Application can still be used
    # Doesnt work because Matplot cant Plot in 2nd Thread
    def analyze_thread_start(self):
        my_thread = threading.Thread(target=self.analyze)
        my_thread.start()

    def analyze(self):

        print(self.many_images_paths)

        if self.model.currentText() == "VGG16":
            import keras.applications.vgg16 as vgg16
            # Keras Model
            self.keras_model = vgg16.VGG16(weights="imagenet")
            self.keras_preprocess = vgg16.preprocess_input
            self.keras_decode = vgg16.decode_predictions
            self.last_conv_layer = "block5_conv3"
        elif self.model.currentText() == "VGG19":
            pass
        elif self.model.currentText() == "Custom":
            pass

        if self.analyze_mode.currentText() == "Single Image":
            if self.single_image_path != "":
                self.single_image_analyzer()
        elif self.analyze_mode.currentText() == "Many Images":
            if self.many_images_paths != []:
                self.many_images_analyzer()
        elif self.analyze_mode.currentText() == "Video":
            pass

    def image_analyzer(self, image, image_path):
        image_list = []
        title_list = []
        cmap_list = []
        if self.model.currentText() == "VGG16":
            size = (224, 224)
            resized_image = cv.resize(image, size)
        elif self.model.currentText() == "VGG19":
            pass
        elif self.model.currentText() == "ResNet":
            pass
        noise_walk_value = 0
        if self.noise_walk_checkbox.isChecked():

            max_value = self.noise_walk_max.value()
            steps = self.noise_walk_steps.value()
            stepsize = int(max_value / steps)
            steps = steps + 1
            print(f"steps: {steps}")
        else:
            steps = 1
            stepsize = 0
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

        for i in range(steps):
            if self.noise_walk_checkbox.isChecked():
                resized_image = IMAGE_EDIT.add_noise(resized_image, noise_walk_value)
                image_list.append(resized_image)
                title_list.append(f"Noise Level: {noise_walk_value}")
                cmap_list.append("viridis")
                cv.imwrite("data/noise_image.png", resized_image)

            if self.lrp_checkbox.isChecked():
                rule = self.lrp_rule_box.currentText()
                lrp_image = self.lrp_analyze(resized_image, rule)

                title_list.append(f"LRP: {rule}")
                cmap_list.append('viridis')
                image_list.append(lrp_image)

            if self.gradcam_checkbox.isChecked():
                print(self.grad_cam_version_combobox.currentText())
                if self.grad_cam_version_combobox.currentText() == "GradCam":
                    grad_cam_image = self.grad_cam_analyze(image_path)
                    title_list.append(f"GRAD CAM")
                elif self.grad_cam_version_combobox.currentText() == "GradCam++":
                    grad_cam_image = self.grad_cam_plus_analyze(image_path)
                    title_list.append(f"GRAD CAM++")

                cmap_list.append('viridis')
                image_list.append(grad_cam_image)

            if self.lime_checkbox.isChecked():
                # Samples must be ~50+
                if self.heatmap_box_lime.isChecked():
                    samples = self.lime_samples_box.value()
                    features = "All"
                    lime_image = self.lime_heatmap(resized_image, samples)
                else:
                    samples = self.lime_samples_box.value()
                    features = self.lime_features_box.value()
                    lime_image = self.lime_analyzer(resized_image, samples, features)

                title_list.append(f"LIME - Samples: {samples}, Features: {features}")
                cmap_list.append('viridis')
                image_list.append(lime_image)

            if self.monte_carlo_checkbox.isChecked():
                mcd_samples = self.mcd_samples_box.value()
                mcd_apply_skip = self.MCD_ApplyOrSkip_comboBox.value()
                mcd_layers = self.mcd_Layer_comboBox.Value()
                mcd_dropoutrate = self.mcd_percent_spinBox.Value()
                
                mcd_prediction = self.mcDroupout(self, image, mcd_samples, mcd_dropoutrate, mcd_apply_skip, mcd_layers)
                
                #not there jet
                #outputmcdList.append(mcd_prediction)

            if self.overlap_box.isChecked():
                overlap_image = self.overlap_images(image_list)
                print("Overlap Image")
                overlap_image = convert_to_uint8(overlap_image)
                title_list.append('Overlap IMAGE')
                image_list.append(overlap_image)
                cmap_list.append('viridis')

            noise_walk_value = noise_walk_value + stepsize

        return image_list, cmap_list, title_list

    def single_image_analyzer(self):
        image = cv.imread(self.single_image_path, cv.IMREAD_COLOR)
        image_list, cmap_list, title_list = self.image_analyzer(image, self.single_image_path)
        print("Plotting")
        length = self.find_len_per_row()
        rows = len(image_list) / length
        PLOTTING.plot_n_images(image_list, title_list, cmap_list,
                               max_images_per_row=self.find_len_per_row(),
                               figsize=(length * 5, rows * 5))

    def many_images_analyzer(self):
        image_list = []
        title_list = []
        cmap_list = []

        for i in range(len(self.many_images_paths)):
            image = cv.imread(self.many_images_paths[i], cv.IMREAD_COLOR)
            print(f"INDEX: {i}")
            temp_image_list, temp_cmap_list, temp_title_list = self.image_analyzer(image, self.many_images_paths[i])
            print(f"INDEX: {i} - durchgelaufen")
            image_list = image_list + temp_image_list
            title_list = title_list + temp_title_list
            cmap_list = cmap_list + temp_cmap_list
        print("Plotting")

        length = self.find_len_per_row()
        rows = len(image_list) / length
        PLOTTING.plot_n_images(image_list, title_list, cmap_list,
                               max_images_per_row=self.find_len_per_row(),
                               figsize=(length * 5, rows * 5))

    def lrp_analyze(self, image, rule):
        print("LRP")

        lrp_image = LRP.analyze_image_lrp(image, self.keras_model, self.keras_preprocess, rule)
        lrp_image = convert_to_uint8(lrp_image)
        zeros_array = np.zeros_like(lrp_image)
        lrp_image = cv.merge([zeros_array, zeros_array, lrp_image])  # LRP nur im roten Kanal
        return lrp_image

    def find_len_per_row(self):
        len_per_row = 1
        if not self.noise_walk_checkbox.isChecked():
            if self.noise_checkbox.isChecked():
                len_per_row += 1
        if self.lrp_checkbox.isChecked():
            len_per_row += 1
        if self.gradcam_checkbox.isChecked():
            len_per_row += 1
        if self.lime_checkbox.isChecked():
            len_per_row += 1
        if self.overlap_box.isChecked():
            len_per_row += 1
        return len_per_row

    def grad_cam_analyze(self, image_path):
        # TODO Pickle Model Übergeben, Last Conv Layer
        print("GRAD_CAM")
        import subprocess
        # Specify the path to the TensorFlow script
        tensorflow_script_path = "CAM/framework_grad_cam.py"
        # Specify the model_name and filepath as arguments
        model_name = self.model.currentText()
        if self.noise_checkbox.isChecked() or self.noise_walk_checkbox.isChecked():
            filepath = 'data/noise_image.png'
        else:
            filepath = image_path
        # Run the TensorFlow script as a subprocess with arguments
        subprocess.run(["python", tensorflow_script_path, model_name, filepath])
        grad_cam_image = cv.imread("data/grad_cam.jpg")
        grad_cam_image = convert_to_uint8(grad_cam_image)
        return grad_cam_image
    
    def grad_cam_plus_analyze(self, image_path):
        # TODO Pickle Model Übergeben, Last Conv Layer
        print("GRAD_CAM_++")
        import subprocess
        # Specify the path to the TensorFlow script
        tensorflow_script_path = "CAM/framework_gcam_plus.py"
        # Specify the model_name and filepath as arguments
        model_name = self.model.currentText()
        if self.noise_checkbox.isChecked() or self.noise_walk_checkbox.isChecked():
            filepath = 'data/noise_image.png'
        else:
            filepath = image_path
        # Run the TensorFlow script as a subprocess with arguments
        subprocess.run(["python", tensorflow_script_path, model_name, filepath])
        grad_cam_image = cv.imread("data/grad_cam_plusplus.jpg")
        grad_cam_image = convert_to_uint8(grad_cam_image)
        return grad_cam_image

    def lime_analyzer(self, image, samples, features):
        print("LIME_ANALYZER")
        lime_image = LIME.get_lime_explanation(image, self.keras_model, samples, features)
        lime_image = convert_to_uint8(lime_image)
        return lime_image

    def lime_heatmap(self, image, samples):
        print("LIME_HEATMAP")
        heatmap = LIME.get_lime_heat_map(image, self.keras_model, samples)
        heatmap = convert_to_uint8(heatmap)
        zeros_array = np.zeros_like(heatmap)
        heatmap = cv.merge([heatmap, zeros_array, zeros_array])  # LIME Heatmap nur blau
        return heatmap
    
    def mcDropout(self, image, mcd_samples, mcd_dropoutrate, mcd_apply_skip, mcd_layers):
        print("MCD_PREDICTION")
        mcd_prediction = mcd.get_mcd_uncertainty(image, self.keras_model, self.keras_preprocess, self.keras_decode, mcd_samples, mcd_dropoutrate, mcd_apply_skip, mcd_layers)
        return mcd_prediction

    def overlap_images(self, image_list):
        print("OVERLAP")
        if len(image_list) > 1:
            length = self.find_len_per_row()
            overlap_images = image_list[-length:]
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
with open('ui.qss', 'r') as styles_file:
    qss = styles_file.read()
app.setStyleSheet(qss)
window = Ui()
app.exec_()

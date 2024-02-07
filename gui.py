from PyQt5 import uic
import sys
import cv2 as cv
import LRP.LRP as LRP
import LIME.LIME as LIME
from custom import custom_model as custom_model
from MonteCarloDropout import mcd
from utils import datei_laden, IMAGE_EDIT, PLOTTING
import numpy as np
import tensorflow as tf
import threading
from VideoPlayer import VideoPlayer
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from PyQt5.QtGui import QIcon

tf.compat.v1.disable_eager_execution()
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (qApp, QFileDialog,
                             QListWidgetItem)
import os

import keras


class AnotherWindowGame(QWidget, QtCore.QThread):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)
        my_thread2 = threading.Thread(target=self.runGame)
        my_thread2.start()

    def runGame(self):
        from GAME.chromedino import menu
        menu(0)


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
        self.img_size = (0, 0)
        self.custom_channels = 3
        # Custom Model
        self.custom_model_path = ''
        self.custom_model_weights_path = ''
        self.custom_model_mapping_path = ''
        # Loaded Files
        self.single_image_path = ""
        self.many_images_paths = []
        self.video_path = ""

        # Initialize the video player
        self.video_player = VideoPlayer()
        self.video_layout.addWidget(self.video_player)
        
        # Set Icon
        icon = QIcon("data\og.png")  # Replace with the path to your icon
        self.setWindowIcon(icon)

        # Buttons
        self.button_load_single_image.clicked.connect(self.file_dialog_single)
        self.button_load_many_images.clicked.connect(self.file_dialog_many)
        self.button_load_video.clicked.connect(self.file_dialog_video)
        self.button_analyze.clicked.connect(self.analyze_thread_start)
        self.tutorial_button.clicked.connect(self.show_tutorial_window)

        self.load_model_button.clicked.connect(lambda: self.file_open_dialog_model('model'))
        self.load_model_weights_button.clicked.connect(lambda: self.file_open_dialog_model('weights'))
        self.mapping_button.clicked.connect(lambda: self.file_open_dialog_model('mapping'))

        # Connect the QListWidget
        self.image_list_widget.itemClicked.connect(self.show_selected_image)

        self.show()

    def file_open_dialog_model(self, file_type):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setViewMode(QFileDialog.ViewMode.List)
        if file_type == 'model':
            dialog.setNameFilter("Model (*.keras *.h5)")
        elif file_type == 'weights':
            dialog.setNameFilter("Model (*.keras *.h5)")
        elif file_type == 'mapping':
            dialog.setNameFilter("Mapping (*.csv)")

        if dialog.exec():
            filename = dialog.selectedFiles()
            if file_type == 'model':
                dialog.setNameFilter("Model (*.keras *.h5)")
                self.custom_model_path = filename[0]
                self.model_text.setText(os.path.basename(self.custom_model_path))
            elif file_type == 'weights':
                dialog.setNameFilter("Model (*.keras *.h5)")
                self.custom_model_weights_path = filename[0]
                self.model_weights_text.setText(os.path.basename(self.custom_model_weights_path))
            elif file_type == 'mapping':
                dialog.setNameFilter("Mapping (*.csv)")
                self.custom_model_mapping_path = filename[0]
                self.model_mapping_text.setText(os.path.basename(self.custom_model_mapping_path))

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
        import os
        os_name = os.name
        if os_name == 'nt':  # Windows
            self.analyze()
        else:
            my_thread = threading.Thread(target=self.analyze)
            my_thread.start()

    def analyze(self):

        if (self.Game.isChecked()):
            self.show_new_window()

        print(self.many_images_paths)

        if self.model.currentText() == "VGG16":
            import keras.applications.vgg16 as vgg16
            # Keras Model
            self.keras_model = vgg16.VGG16(weights="imagenet")
            self.keras_preprocess = vgg16.preprocess_input
            self.keras_decode = vgg16.decode_predictions
            self.img_size = (self.keras_model.input_shape[1], self.keras_model.input_shape[2])
            self.custom_channels = self.keras_model.input_shape[3]
        elif self.model.currentText() == "VGG19":
            import keras.applications.vgg19 as vgg19
            # Keras Model
            self.keras_model = vgg19.VGG19(weights="imagenet")
            self.keras_preprocess = vgg19.preprocess_input
            self.keras_decode = vgg19.decode_predictions
            self.img_size = (self.keras_model.input_shape[1], self.keras_model.input_shape[2])
            self.custom_channels = self.keras_model.input_shape[3]
        else:
            custom_model.set_csv_file_path(self.custom_model_mapping_path)

            self.keras_model = keras.models.load_model(self.custom_model_path)
            self.keras_model.load_weights(self.custom_model_weights_path)
            self.keras_decode = custom_model.decode_predictions
            self.keras_preprocess = custom_model.preprocess
            self.img_size = (self.keras_model.input_shape[1], self.keras_model.input_shape[2])
            self.custom_channels = self.keras_model.input_shape[3]
            custom_model.set_size(self.img_size)
            custom_model.set_channels(self.custom_channels)

        if self.analyze_mode.currentText() == "Single Image":
            if self.single_image_path != "":
                self.single_image_analyzer()
        elif self.analyze_mode.currentText() == "Many Images":
            if self.many_images_paths != []:
                self.many_images_analyzer()
        elif self.analyze_mode.currentText() == "Video":
            if self.video_path != "":
                self.video_analyzer()

    def video_analyzer(self):
        video_path = self.video_path
        if self.videoComboBox.currentText() == "GradCam":
            analyzed_video_path = self.grad_cam_video_analyze(video_path)
        elif self.videoComboBox.currentText() == "GradCam++":
            analyzed_video_path = self.gcam_plus_video_analyze(video_path)
        # Create a QDialog for the video player pop-up
        videoPlayerDialog = QtWidgets.QDialog(self)
        videoPlayerDialog.setWindowTitle('Video Player')
        videoPlayerDialog.setGeometry(100, 100, 800, 800)
        videoPlayerDialog.setMinimumSize(400, 400)

        # Create an instance of the VideoPlayer widget
        videoPlayer = VideoPlayer()
        # Initialize the video player
        video_layout = QVBoxLayout()
        video_layout.addWidget(videoPlayer)
        video_layout.addWidget(videoPlayer.videoWidget)

        # Button to close the video player pop-up
        closeDialogButton = QtWidgets.QPushButton('Close')
        closeDialogButton.clicked.connect(videoPlayerDialog.close)

        # Layout for the video player pop-up
        dialogLayout = QVBoxLayout()
        dialogLayout.addLayout(video_layout)
        dialogLayout.addWidget(closeDialogButton)

        videoPlayerDialog.setLayout(dialogLayout)

        # Load a video into the VideoPlayer
        videoPlayer.load_video(analyzed_video_path)

        # Show the video player pop-up
        videoPlayerDialog.exec_()

    def image_analyzer(self, image, image_path):
        image_list = []
        title_list = []
        cmap_list = []

        # if self.model.currentText() == "VGG16":
        #     self.img_size = (224, 224)
        #     resized_image = cv.resize(image, self.img_size)
        # elif self.model.currentText() == "VGG19":
        #     self.img_size = (224, 224)
        #     resized_image = cv.resize(image, self.img_size)
        # elif self.model.currentText() == "ResNet50":
        #     self.img_size = (224, 224)
        #     resized_image = cv.resize(image, self.img_size)
        # else:
        #     self.img_size = (self.custom_model_size_x.value(), self.custom_model_size_y.value())
        #
        resized_image = cv.resize(image, self.img_size)
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
            decoded_pred = pred_get_string(self.keras_model, self.keras_decode, resized_image)
            title_list.append(f"Original {decoded_pred[0][0][1]} - {round(float(decoded_pred[0][0][2]),2)}")
            cmap_list.append("viridis")
            if self.noise_checkbox.isChecked():
                noise_level = self.noiselevel_box.value()
                resized_image = IMAGE_EDIT.add_noise(resized_image, noise_level)

                image_list.append(resized_image)

                decoded_pred = pred_get_string(self.keras_model, self.keras_decode, resized_image)
                title_list.append(f"Noise {decoded_pred[0][0][1]} - {round(float(decoded_pred[0][0][2]),2)}")

                cmap_list.append("viridis")
                cv.imwrite("data/noise_image.png", resized_image)

        for i in range(steps):
            if self.noise_walk_checkbox.isChecked():
                resized_image = IMAGE_EDIT.add_noise(resized_image, noise_walk_value)
                image_list.append(resized_image)
                decoded_pred = pred_get_string(self.keras_model, self.keras_decode, resized_image)
                title_list.append(f"Noise Level {noise_walk_value} {decoded_pred[0][0][1]} - {round(float(decoded_pred[0][0][2]),2)}")
                cmap_list.append("viridis")
                cv.imwrite("data/noise_image.png", resized_image)

            if self.lrp_checkbox.isChecked():
                rule = self.lrp_rule_box.currentText()
                lrp_image = self.lrp_analyze(resized_image, rule)
                lrp_image = convert_to_uint8(lrp_image)
                title_list.append(f"LRP: {rule}")
                cmap_list.append('viridis')
                image_list.append(lrp_image)

            if self.gradcam_checkbox.isChecked():
                print(self.grad_cam_version_combobox.currentText())
                if self.grad_cam_version_combobox.currentText() == "GradCam":
                    grad_cam_image, grad_cam_heatmap = self.grad_cam_analyze(image_path)
                    if self.heatmap_box_cam.isChecked():
                        image_list.append(grad_cam_heatmap)
                        title_list.append(f"GRAD CAM Heatmap")
                    else:
                        image_list.append(grad_cam_image)
                        title_list.append(f"GRAD CAM")

                elif self.grad_cam_version_combobox.currentText() == "GradCam++":
                    grad_cam_image, grad_cam_heatmap = self.grad_cam_plus_analyze(image_path)
                    if self.heatmap_box_cam.isChecked():
                        image_list.append(grad_cam_heatmap)
                        title_list.append(f"GRAD CAM++ Heatmap")
                    else:
                        image_list.append(grad_cam_image)
                        title_list.append(f"GRAD CAM++")

                cmap_list.append('viridis')

            if self.lime_checkbox.isChecked():
                # Samples must be ~50+
                if self.heatmap_box_lime.isChecked():
                    samples = self.lime_samples_box.value()
                    features = "All"
                    lime_image = self.lime_heatmap(resized_image, samples)
                    title_list.append(f"LIME Heatmap - Samples: {samples}, Features: {features}")
                else:
                    if self.lime_positive_only_box.isChecked():
                        positive_only = True
                    else:
                        positive_only = False
                    if self.lime_hide_rest_box.isChecked():
                        hide_rest = True
                    else:
                        hide_rest = False
                    if self.lime_mask_only_box.isChecked():
                        mask_only = True
                    else:
                        mask_only = False
                    if self.lime_min_weight_box.isChecked():
                        min_weight = self.lime_min_weight_box_2.value()
                    else:
                        min_weight = 0

                    samples = self.lime_samples_box.value()
                    features = self.lime_features_box.value()
                    lime_image = self.lime_analyzer(resized_image, samples, features, positive_only, hide_rest, mask_only, min_weight)
                    title_list.append(f"LIME - Samples: {samples}, Features: {features}")

                cmap_list.append('viridis')
                image_list.append(lime_image)

            if self.overlap_box.isChecked():
                overlap_image = self.overlap_images(image_list, title_list)
                print("Overlap Image")
                overlap_image = convert_to_uint8(overlap_image)
                title_list.append('Overlap IMAGE')
                image_list.append(overlap_image)
                cmap_list.append('viridis')

            if self.monte_carlo_checkbox.isChecked():
                mcd_samples = self.mcd_samples_box.value()
                mcd_apply_skip = self.mcd_apply_skip_comboBox.currentText()
                mcd_dropoutrate = self.mcd_percent_spinBox.value()
                mcd_LayerList = self.mcd_create_layer_list()

                mcd_prediction = self.mcDropout(resized_image, mcd_samples, mcd_dropoutrate, mcd_apply_skip,
                                                mcd_LayerList)
                print(mcd_prediction)
                string_samples = str(mcd_samples) + " sample(s)"
                print(string_samples)
                string_dropout = "Dropout: " + str(mcd_dropoutrate) + "%"
                print(string_dropout)
                string_prediction1 = "TOP1 MCD = " + str(float(mcd_prediction[0])) + "% " + str(
                    mcd_prediction[1])
                float_prediction1 = float(mcd_prediction[0])
                print(string_prediction1)
                string_prediction2 = "TOP2 MCD = " + str(float(mcd_prediction[2])) + "% " + str(
                    mcd_prediction[3])
                float_prediction2 = float(mcd_prediction[2])
                print(string_prediction2)
                string_prediction3 = "TOP3 MCD = " + str(float(mcd_prediction[4])) + "% " + str(
                    mcd_prediction[5])
                float_prediction3 = float(mcd_prediction[4])
                print(string_prediction3)
                string_prediction4 = "TOP4 MCD = " + str(float(mcd_prediction[6])) + "% " + str(
                    mcd_prediction[7])
                float_prediction4 = float(mcd_prediction[6])
                print(string_prediction4)
                string_prediction5 = "TOP5 MCD = " + str(float(mcd_prediction[8])) + "% " + str(
                    mcd_prediction[9])
                float_prediction5 = float(mcd_prediction[8])
                print(string_prediction5)
                mcd_image = create_mcd_image(string_samples, string_dropout, string_prediction1, float_prediction1,
                                             string_prediction2, float_prediction2, string_prediction3, float_prediction3, string_prediction4, float_prediction4,
                                             string_prediction5, float_prediction5)
                print(string_samples + string_dropout + string_prediction1)
                image_list.append(mcd_image)
                cmap_list.append('viridis')
                title_list.append("Monte Carlo")

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
        if self.monte_carlo_checkbox.isChecked():
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
        tensorflow_script_path = "framework_grad_cam.py"
        # Specify the model_name and filepath as arguments
        model_name = self.model.currentText()
        if self.noise_checkbox.isChecked() or self.noise_walk_checkbox.isChecked():
            filepath = 'data/noise_image.png'
        else:
            filepath = image_path
        # Run the TensorFlow script as a subprocess with arguments
        import json
        json_img_size = json.dumps(self.img_size)
        subprocess.run(["python", tensorflow_script_path, model_name, filepath, json_img_size,
                        self.custom_model_path, self.custom_model_weights_path, self.custom_model_mapping_path,
                        str(self.custom_channels)])

        grad_cam_image = cv.imread("data\gradcam_output\Large_Heatmap\cam1_3.jpg")
        grad_cam_image = convert_to_uint8(grad_cam_image)
        grad_cam_image = cv.resize(grad_cam_image, self.img_size)

        grad_cam_heatmap = cv.imread("data\gradcam_output\Mid_Heatmap\cam1_2.jpg")
        grad_cam_heatmap = convert_to_uint8(grad_cam_heatmap)
        grad_cam_heatmap = cv.resize(grad_cam_heatmap, self.img_size)

        return grad_cam_image, grad_cam_heatmap

    def grad_cam_plus_analyze(self, image_path):
        # TODO Pickle Model Übergeben, Last Conv Layer
        print("GRAD_CAM++")
        import subprocess
        # Specify the path to the TensorFlow script
        tensorflow_script_path = "framework_gcam_plus.py"
        # Specify the model_name and filepath as arguments
        model_name = self.model.currentText()
        if self.noise_checkbox.isChecked() or self.noise_walk_checkbox.isChecked():
            filepath = 'data/noise_image.png'
        else:
            filepath = image_path
        # Run the TensorFlow script as a subprocess with arguments
        import json
        json_img_size = json.dumps(self.img_size)
        subprocess.run(["python", tensorflow_script_path, model_name, filepath, json_img_size,
                        self.custom_model_path, self.custom_model_weights_path, self.custom_model_mapping_path,
                        str(self.custom_channels)])

        grad_cam_image = cv.imread("data\gcam_plus_output\Large_Heatmap\cam2_3.jpg")
        grad_cam_image = convert_to_uint8(grad_cam_image)
        grad_cam_image = cv.resize(grad_cam_image, self.img_size)

        grad_cam_heatmap = cv.imread("data\gcam_plus_output\Mid_Heatmap\cam2_2.jpg")
        grad_cam_heatmap = convert_to_uint8(grad_cam_heatmap)
        grad_cam_heatmap = cv.resize(grad_cam_heatmap, self.img_size)

        return grad_cam_image, grad_cam_heatmap

    def grad_cam_video_analyze(self, video_path):
        # TODO Pickle Model Übergeben, Last Conv Layer
        print("GRAD_CAM_Video")
        import subprocess
        # Specify the path to the TensorFlow script
        tensorflow_script_path = "framework_grad_video.py"
        # Specify the model_name and filepath as arguments
        model_name = self.model.currentText()
        filepath = video_path
        # Run the TensorFlow script as a subprocess with arguments
        import json
        json_img_size = json.dumps(self.img_size)
        subprocess.run(["python", tensorflow_script_path, model_name, filepath, json_img_size,
                        self.custom_model_path, self.custom_model_weights_path, self.custom_model_mapping_path,
                        str(self.custom_channels)])
        grad_cam_video = "data\gradcam_output\LH_video.avi"
        return grad_cam_video
    
    def gcam_plus_video_analyze(self, video_path):
        # TODO Pickle Model Übergeben, Last Conv Layer
        print("GRAD_CAM++_Video")
        import subprocess
        # Specify the path to the TensorFlow script
        tensorflow_script_path = "framework_gplus_video.py"
        # Specify the model_name and filepath as arguments
        model_name = self.model.currentText()
        filepath = video_path
        # Run the TensorFlow script as a subprocess with arguments
        import json
        json_img_size = json.dumps(self.img_size)
        subprocess.run(["python", tensorflow_script_path, model_name, filepath, json_img_size,
                        self.custom_model_path, self.custom_model_weights_path, self.custom_model_mapping_path,
                        str(self.custom_channels)])
        grad_cam_video = "data\gcam_plus_output\LH_video.avi"
        return grad_cam_video

    def lime_analyzer(self, image, samples, features, positive_only, hide_rest, mask_only, min_weight):
        print("LIME_ANALYZER")
        lime_image = LIME.get_lime_explanation(image, self.keras_model, samples, features, self.keras_preprocess, 
                                               positive_only, hide_rest, mask_only, min_weight)
        lime_image = convert_to_uint8(lime_image)
        return lime_image

    def lime_heatmap(self, image, samples):
        print("LIME_HEATMAP")
        heatmap = LIME.get_lime_heat_map(image, self.keras_model, samples, self.keras_preprocess)
        heatmap = convert_to_uint8(heatmap)
        zeros_array = np.zeros_like(heatmap)
        heatmap = cv.merge([zeros_array, heatmap, zeros_array])  # LIME Heatmap nur blau
        return heatmap

    def mcDropout(self, image, mcd_samples, mcd_dropoutrate, mcd_apply_skip, mcd_layers):
        print("MCD_PREDICTION")
        mcd_prediction = mcd.get_mcd_uncertainty(image, self.keras_model, self.keras_preprocess, self.keras_decode,
                                                 mcd_samples, mcd_dropoutrate, mcd_apply_skip, mcd_layers)
        return mcd_prediction

    def show_new_window(self):
        AnotherWindowGame()

    def show_tutorial_window(self):
        # Load the help dialog from the .ui file
        help_dialog = QtWidgets.QDialog(self)
        uic.loadUi('tutorial.ui', help_dialog)

        # Add any additional customization or connections here
        help_dialog.exec_()
        
    
    def mcd_create_layer_list(self):
        mcd_layers = []
        if (self.mcd_activation_radio.isChecked):
            mcd_layers.append("ReLU")
            mcd_layers.append("Softmax")
            mcd_layers.append("LeakyReLU")
            mcd_layers.append("PReLU")
            mcd_layers.append("ELU")
        if (self.mcd_batch_norm_radio.isChecked):
            mcd_layers.append("BatchNormalization")
        if (self.mcd_convolutional_radio.isChecked):
            mcd_layers.append("conv")
        if (self.mcd_dense_radio.isChecked):
            mcd_layers.append("Dense")
        if (self.mcd_group_norm_radio.isChecked):
            mcd_layers.append("GroupNormalization")
        if (self.mcd_layer_norm_radio.isChecked):
            mcd_layers.append("LayerNormalization")
        if (self.mcd_unit_norm_radio.isChecked):
            mcd_layers.append("UnitNormalization")
        return mcd_layers

    def overlap_images(self, image_list, title_list):
        print("OVERLAP")
        overlap_images = []
        for i, title in enumerate(title_list):
            if "lrp" in title.lower():
                overlap_images.append(image_list[i] * 2)
            elif "lime heatmap" in title.lower():
                overlap_images.append(image_list[i] // 2)  # Heatmaps sind zu dominant man sieht lrp nicht mehr
            elif "grad cam heatmap" in title.lower():
                overlap_images.append(image_list[i] // 2)
        overlap_image = IMAGE_EDIT.overlap_images(overlap_images)
        return overlap_image
    def overlap_images_try(self, image_list, title_list):
        print("OVERLAP New")
        red_image = cv.cvtColor(np.zeros_like(image_list[0]), cv.COLOR_BGR2GRAY)
        green_image = red_image
        blue_image = red_image
        for i, title in enumerate(title_list):
            if "lrp" in title.lower():
                red_image = cv.cvtColor(image_list[i], cv.COLOR_BGR2GRAY) * 3
            elif "lime heatmap" in title.lower():
                green_image = cv.cvtColor(image_list[i], cv.COLOR_BGR2GRAY)
            elif "grad cam heatmap" in title.lower():
                blue_image = cv.cvtColor(image_list[i], cv.COLOR_BGR2GRAY)
        overlap_image = cv.merge([blue_image, green_image, red_image])
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

def pred_get_string(model, decode, image):
    preds = model.predict(image[None])
    decoded_pred = decode(preds, top=1)
    return decoded_pred


def create_mcd_image(text1, text2, pred1, float_pred1, pred2, float_pred2, pred3, float_pred3, pred4, float_pred4,
                     pred5, float_pred5):
    from PIL import Image, ImageDraw, ImageFont
    # Set image dimensions
    width, height = (224, 224)

    # Create a white background image
    image = Image.new("RGB", (width, height), "white")

    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Set font properties
    font_size = 10
    # font = ImageFont.truetype("arial.ttf", font_size)  # Use a suitable font file path
    font = ImageFont.load_default()

    # Set text positions
    x1, y1 = int(width * 0), int(height * 0.2)
    x2, y2 = int(width * 0), int(height * 0.3)
    x3, y3 = int(width * 0), int(height * 0.4)
    x4, y4 = int(width * 0), int(height * 0.5)
    x5, y5 = int(width * 0), int(height * 0.6)
    x6, y6 = int(width * 0), int(height * 0.8)
    x7, y7 = int(width * 0), int(height * 0.9)

    # Draw a linePlot
    draw.line((x1, y1, x1 + int(width / 100 * float_pred1), y1), fill=("red"), width=int(height / 10))
    draw.line((x2, y2, x2 + int(width / 100 * float_pred2), y2), fill=("red"), width=int(height / 10))
    draw.line((x3, y3, x3 + int(width / 100 * float_pred3), y3), fill=("red"), width=int(height / 10))
    draw.line((x4, y4, x4 + int(width / 100 * float_pred4), y4), fill=("red"), width=int(height / 10))
    draw.line((x5, y5, x5 + int(width / 100 * float_pred5), y5), fill=("red"), width=int(height / 10))

    # Draw black text on the image
    draw.text((x1, y1), pred1, font=font, fill="black")
    draw.text((x2, y2), pred2, font=font, fill="black")
    draw.text((x3, y3), pred3, font=font, fill="black")
    draw.text((x4, y4), pred4, font=font, fill="black")
    draw.text((x5, y5), pred5, font=font, fill="black")
    draw.text((x6, y6), text1 + " " + text2, font=font, fill="black")

    # Convert the PIL Image to a NumPy array
    image_array = np.array(image)
    image_array = convert_to_uint8(image_array)
    return image_array


app = QtWidgets.QApplication(sys.argv)
with open('ui.qss', 'r') as styles_file:
    qss = styles_file.read()
app.setStyleSheet(qss)
window = Ui()
app.exec_()

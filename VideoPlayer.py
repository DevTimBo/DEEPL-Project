# Autor: Jason Pranata
# Date: 13 February 2024 
# Description: This file contains the VideoPlayer class that will be used in the main file to play the video in the framework

from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QFileDialog
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    # Initialize the UI
    def initUI(self):
        self.videoWidget = QVideoWidget()
        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.videoWidget)

        # Button
        self.playPauseButton = QPushButton('Play/Pause')
        self.playPauseButton.clicked.connect(self.on_playPauseButton_clicked)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.videoWidget)
        layout.addWidget(self.playPauseButton)

        self.setLayout(layout)

    # function to load video
    def load_video(self, video_path):
        media_url = QUrl.fromLocalFile(video_path)
        content = QMediaContent(media_url)
        self.player.setMedia(content)
        self.player.play()

    # function to play/pause video for the button
    def on_playPauseButton_clicked(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

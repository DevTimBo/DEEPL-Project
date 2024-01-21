# video_player.py
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QFileDialog
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

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

    def load_video(self, video_path):
        media_url = QUrl.fromLocalFile(video_path)
        content = QMediaContent(media_url)
        self.player.setMedia(content)
        self.player.play()

    def on_playPauseButton_clicked(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

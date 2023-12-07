#! /usr/bin/env python3
#
# PyQt5 example for VLC Python bindings
# Copyright (C) 2009-2010 the VideoLAN team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston MA 02110-1301, USA.
#
"""
A simple example for VLC python bindings using PyQt5.

Author: Saveliy Yusufov, Columbia University, sy2685@columbia.edu
Date: 25 December 2018
"""

import platform
import os
import sys

from datetime import timedelta
from PyQt5 import QtWidgets, QtGui, QtCore
import vlc


class Player(QtWidgets.QMainWindow):
    """A simple Media Player using VLC and Qt
    """

    def __init__(self, filepath, query_time, total_duration, master=None):
        QtWidgets.QMainWindow.__init__(self, master)
        self.setWindowTitle("Media Player")

        # Create a basic vlc instance
        self.instance = vlc.Instance()
        self.filepath = filepath

        self.media = None

        # Create an empty vlc media player
        self.mediaplayer = self.instance.media_player_new()
        self.total_duration = int(total_duration)
        self.query_time = query_time


        self.create_ui()
        self.is_paused = False

    def create_ui(self):
        """Set up the user interface, signals & slots
        """
        self.widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.widget)

        # In this widget, the video will be drawn
        if platform.system() == "Darwin":  # for MacOS
            self.videoframe = QtWidgets.QMacCocoaViewContainer(0)
        else:
            self.videoframe = QtWidgets.QFrame()

        self.palette = self.videoframe.palette()
        self.palette.setColor(QtGui.QPalette.Window, QtGui.QColor(0, 0, 0))
        self.videoframe.setPalette(self.palette)
        self.videoframe.setAutoFillBackground(True)

        self.sliderLayout = QtWidgets.QHBoxLayout(self)
        self.positionslider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.positionslider.setToolTip("Position")
        self.positionslider.setMaximum(1000)
        self.positionslider.sliderMoved.connect(self.set_position)
        self.positionslider.sliderPressed.connect(self.set_position)
        self.positionslider.setFixedHeight(20)

        self.running_time_label = QtWidgets.QLabel("Hello", self)
        self.running_time_label.setFixedHeight(20)
        self.running_time_label.setFixedWidth(50)
        minutes = self.total_duration // 60
        seconds = self.total_duration % 60
        self.total_duration_label = QtWidgets.QLabel(str(minutes) + ":" + str(seconds))
        self.total_duration_label.setFixedHeight(20)
        self.sliderLayout.addWidget(self.running_time_label)
        self.sliderLayout.addWidget(self.positionslider)
        self.sliderLayout.addWidget(self.total_duration_label)

        self.hbuttonbox = QtWidgets.QHBoxLayout()
        self.playbutton = QtWidgets.QPushButton("Play")
        self.hbuttonbox.addWidget(self.playbutton)
        self.playbutton.clicked.connect(self.play_pause)

        self.stopbutton = QtWidgets.QPushButton("Stop")
        self.hbuttonbox.addWidget(self.stopbutton)
        self.stopbutton.clicked.connect(self.stop)

        self.play_from_beginning = QtWidgets.QPushButton("Play From Start")
        self.hbuttonbox.addWidget(self.play_from_beginning)
        self.play_from_beginning.clicked.connect(self.play_from_start)

        self.play_from_query = QtWidgets.QPushButton("Play From Query")
        self.hbuttonbox.addWidget(self.play_from_query)
        self.play_from_query.clicked.connect(self.play_from_query_time)

        self.rewind = QtWidgets.QPushButton("Rewind")
        self.hbuttonbox.addWidget(self.rewind)
        self.rewind.clicked.connect(self.rewind_video)

        self.forward = QtWidgets.QPushButton("Forward")
        self.hbuttonbox.addWidget(self.forward)
        self.forward.clicked.connect(self.fast_forward)

        self.hbuttonbox.addStretch(1)
        self.volumeslider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.volumeslider.setMaximum(100)
        self.volumeslider.setValue(self.mediaplayer.audio_get_volume())
        self.volumeslider.setToolTip("Volume")
        self.hbuttonbox.addWidget(self.volumeslider)
        self.volumeslider.valueChanged.connect(self.set_volume)

        self.vboxlayout = QtWidgets.QVBoxLayout()
        self.vboxlayout.addWidget(self.videoframe)
        self.vboxlayout.addLayout(self.sliderLayout)
        self.vboxlayout.addLayout(self.hbuttonbox)

        self.widget.setLayout(self.vboxlayout)

        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("File")

        # Add actions to file menu
        open_action = QtWidgets.QAction("Load Video", self)
        close_action = QtWidgets.QAction("Close App", self)
        file_menu.addAction(open_action)
        file_menu.addAction(close_action)

        open_action.triggered.connect(self.open_file)
        close_action.triggered.connect(sys.exit)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_ui)

        self.open_file()

    def play_pause(self):
        """Toggle play/pause status
        """
        if self.mediaplayer.is_playing():
            self.mediaplayer.pause()
            self.playbutton.setText("Play")
            self.is_paused = True
            self.timer.stop()
        else:
            if self.mediaplayer.play() == -1:
                self.open_file()
                return

            self.mediaplayer.play()
            self.playbutton.setText("Pause")
            self.timer.start()
            self.is_paused = False

    def stop(self):
        """Stop player
        """
        self.mediaplayer.stop()
        self.playbutton.setText("Play")

    def play_from_start(self):
        self.stop()
        self.media = self.instance.media_new(self.filepath)

        # Put the media in the media player
        self.mediaplayer.set_media(self.media)

        # Parse the metadata of the file
        self.media.parse()
        self.media.add_option('start-time=0.0')
        self.play_pause()

    def play_from_query_time(self):
        self.stop()
        if self.query_time is not None:
            self.media = self.instance.media_new(self.filepath)

            # Put the media in the media player
            self.mediaplayer.set_media(self.media)

            # Parse the metadata of the file
            self.media.parse()
            self.media.add_option('start-time=' + self.query_time)
        self.play_pause()

    def fast_forward(self):
        if self.mediaplayer.is_playing():
            current_time = self.mediaplayer.get_time() + 10000
            self.mediaplayer.set_time(current_time)

    def rewind_video(self):
        if self.mediaplayer.is_playing():
            current_time = self.mediaplayer.get_time() - 10000
            self.mediaplayer.set_time(current_time)

    def open_file(self):
        """Open a media file in a MediaPlayer
        """

        # dialog_txt = "Choose Media File"
        # filename = QtWidgets.QFileDialog.getOpenFileName(self, dialog_txt, os.path.expanduser('~'))
        # if not filename:
        #     return

        # getOpenFileName returns a tuple, so use only the actual file name
        self.media = self.instance.media_new(self.filepath)

        # Put the media in the media player
        self.mediaplayer.set_media(self.media)

        # Parse the metadata of the file
        self.media.parse()

        # Set the title of the track as window title
        self.setWindowTitle(self.media.get_meta(0))

        # The media player has to be 'connected' to the QFrame (otherwise the
        # video would be displayed in it's own window). This is platform
        # specific, so we must give the ID of the QFrame (or similar object) to
        # vlc. Different platforms have different functions for this
        if platform.system() == "Linux":  # for Linux using the X Server
            self.mediaplayer.set_xwindow(int(self.videoframe.winId()))
        elif platform.system() == "Windows":  # for Windows
            self.mediaplayer.set_hwnd(int(self.videoframe.winId()))
        elif platform.system() == "Darwin":  # for MacOS
            self.mediaplayer.set_nsobject(int(self.videoframe.winId()))

        if self.query_time is not None:
            self.media.add_option('start-time=' + self.query_time)
        self.play_pause()

    def set_volume(self, volume):
        """Set the volume
        """
        self.mediaplayer.audio_set_volume(volume)

    def set_position(self):
        """Set the movie position according to the position slider.
        """

        # The vlc MediaPlayer needs a float value between 0 and 1, Qt uses
        # integer variables, so you need a factor; the higher the factor, the
        # more precise are the results (1000 should suffice).

        # Set the media position to where the slider was dragged
        self.timer.stop()
        pos = self.positionslider.value()
        self.mediaplayer.set_position(pos / 1000.0)
        self.timer.start()

    def update_ui(self):
        """Updates the user interface"""

        # Set the slider's position to its corresponding media position
        # Note that the setValue function only takes values of type int,
        # so we must first convert the corresponding media position.
        current_time = self.mediaplayer.get_time() // 1000

        # progress_percentage = (current_time / total_duration) * 100
        # self.progress_bar.set(progress_percentage)
        current_time_str = str(timedelta(seconds=current_time))  # [:-3]
        self.running_time_label.setText(current_time_str)
        # print(current_time_str)
        media_pos = int((current_time / self.total_duration) * 1000)
        # print(media_pos)
        self.positionslider.setValue(media_pos)

        # No need to call this function if nothing is played
        if not self.mediaplayer.is_playing():
            self.timer.stop()

            # After the video finished, the play button stills shows "Pause",
            # which is not the desired behavior of a media player.
            # This fixes that "bug".
            if not self.is_paused:
                self.stop()


def play_video_from(video_path, query_time, total_duration):
    app = QtWidgets.QApplication(sys.argv)
    player = Player(video_path, query_time, total_duration)
    player.show()
    player.resize(640, 480)
    sys.exit(app.exec_())

# def main():
#     """Entry point for our simple vlc player
#     """
#     print(sys.argv[1])
#     app = QtWidgets.QApplication(sys.argv)
#     player = Player(sys.argv[1], 0, sys.argv[2])
#     player.show()
#     player.resize(640, 480)
#     sys.exit(app.exec_())
#
# if __name__ == "__main__":
#     main()
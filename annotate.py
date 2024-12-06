#!/usr/bin/env python3

import sys
from pathlib import Path
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                           QRadioButton, QButtonGroup, QSlider, QPlainTextEdit,
                           QStatusBar)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

class AnnotationTool(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Baseball Event Annotation Tool")

        self.frames = []
        self.frame_index = 0
        self.current_game_id = None
        self.annotations = {}

        self.event_types = {
            0: "no_event",
            1: "home_run",
            2: "out",
            3: "hit"
        }

        self._setup_ui()
        self._create_shortcuts()

    def _setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(10)

        controls = QHBoxLayout()
        self.load_btn = QPushButton("Load Frames")
        self.load_btn.setFixedWidth(200)
        self.save_btn = QPushButton("Save Annotations")
        self.save_btn.setFixedWidth(200)
        self.load_btn.clicked.connect(self.load_frames)
        self.save_btn.clicked.connect(self.save_annotations)
        controls.addWidget(self.load_btn)
        controls.addWidget(self.save_btn)
        layout.addLayout(controls)

        frame_container = QWidget()
        frame_layout = QVBoxLayout(frame_container)
        frame_layout.setContentsMargins(0, 0, 0, 0)

        self.frame_label = QLabel()
        self.frame_label.setMinimumSize(640, 480)
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setStyleSheet("QLabel { background-color: black; }")
        frame_layout.addWidget(self.frame_label)

        nav_layout = QHBoxLayout()
        nav_layout.setContentsMargins(0, 5, 0, 0)

        self.prev_10_btn = QPushButton("◀◀")
        self.prev_btn = QPushButton("◀")
        self.frame_count = QLabel("Frame: 0 / 0")
        self.next_btn = QPushButton("▶")
        self.next_10_btn = QPushButton("▶▶")

        for btn in [self.prev_10_btn, self.prev_btn, self.next_btn, self.next_10_btn]:
            btn.setFixedWidth(100)

        self.prev_10_btn.clicked.connect(lambda: self.jump_frames(-10))
        self.prev_btn.clicked.connect(lambda: self.jump_frames(-1))
        self.next_btn.clicked.connect(lambda: self.jump_frames(1))
        self.next_10_btn.clicked.connect(lambda: self.jump_frames(10))

        self.frame_count.setAlignment(Qt.AlignCenter)
        self.frame_count.setMinimumWidth(120)

        nav_layout.addStretch()
        nav_layout.addWidget(self.prev_10_btn)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.frame_count)
        nav_layout.addWidget(self.next_btn)
        nav_layout.addWidget(self.next_10_btn)
        nav_layout.addStretch()

        frame_layout.addLayout(nav_layout)
        layout.addWidget(frame_container)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.slider_changed)
        layout.addWidget(self.frame_slider)

        event_layout = QHBoxLayout()
        event_layout.setSpacing(20)
        self.event_group = QButtonGroup()

        for event_id, event_name in self.event_types.items():
            radio = QRadioButton(event_name.replace('_', ' ').title())
            self.event_group.addButton(radio, event_id)
            event_layout.addWidget(radio)
            if event_name == "no_event":
                radio.setChecked(True)

        event_layout.addStretch()
        self.mark_btn = QPushButton("Mark Event")
        self.mark_btn.clicked.connect(self.mark_event)
        self.mark_btn.setFixedWidth(120)
        event_layout.addWidget(self.mark_btn)
        layout.addLayout(event_layout)

        self.annotations_display = QPlainTextEdit()
        self.annotations_display.setReadOnly(True)
        self.annotations_display.setMaximumHeight(150)
        layout.addWidget(self.annotations_display)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")

        self.setMinimumSize(800, 700)

    def _create_shortcuts(self):
        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence

        QShortcut(QKeySequence(Qt.Key_Left), self, lambda: self.jump_frames(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self, lambda: self.jump_frames(1))
        QShortcut(QKeySequence(Qt.Key_Space), self, self.mark_event)

        QShortcut(QKeySequence(Qt.CTRL + Qt.Key_Left), self, lambda: self.jump_frames(-10))
        QShortcut(QKeySequence(Qt.CTRL + Qt.Key_Right), self, lambda: self.jump_frames(10))

        for i in range(4):
            QShortcut(QKeySequence(Qt.Key_0 + i), self, 
                     lambda x=i: self.event_group.button(x).setChecked(True))

    def load_frames(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Frames Directory")
        if not directory:
            return

        self.frames = []
        frame_files = sorted(Path(directory).glob('frame_*.jpg'))
        self.current_game_id = Path(directory).name

        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frames.append(frame)

        if self.frames:
            self.frame_index = 0
            self.frame_slider.setRange(0, len(self.frames)-1)
            self.update_display()
            self.statusBar.showMessage(f"Loaded {len(self.frames)} frames from {self.current_game_id}")

    def update_display(self):
        if not self.frames:
            return

        frame = self.frames[self.frame_index]
        height, width = frame.shape[:2]
        scale = min(640/width, 480/height)
        new_width, new_height = int(width * scale), int(height * scale)

        frame = cv2.resize(frame, (new_width, new_height))

        h, w, ch = frame.shape
        qt_image = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.frame_label.setPixmap(QPixmap.fromImage(qt_image))

        self.frame_count.setText(f"Frame: {self.frame_index+1} / {len(self.frames)}")
        self.frame_slider.setValue(self.frame_index)

        self.update_annotations_display()

    def jump_frames(self, delta):
        if not self.frames:
            return

        self.frame_index = max(0, min(self.frame_index + delta, len(self.frames) - 1))
        self.update_display()

    def slider_changed(self, value):
        self.frame_index = value
        self.update_display()

    def mark_event(self):
        if not self.frames or not self.current_game_id:
            return

        event_id = self.event_group.checkedId()
        if event_id == -1:
            return

        if event_id != 0:
            if self.current_game_id not in self.annotations:
                self.annotations[self.current_game_id] = []

            self.annotations[self.current_game_id] = [
                (frame, evt) for frame, evt in self.annotations[self.current_game_id]
                if frame != self.frame_index
            ]

            self.annotations[self.current_game_id].append((self.frame_index, event_id))
            self.annotations[self.current_game_id].sort()

        self.update_annotations_display()
        self.statusBar.showMessage(f"Marked frame {self.frame_index} as {self.event_types[event_id]}")

        self.jump_frames(1)

    def update_annotations_display(self):
        if not self.current_game_id:
            return

        text = f'"{self.current_game_id}": [\n'
        if self.current_game_id in self.annotations:
            annotations = self.annotations[self.current_game_id]
            if annotations:
                text += "    " + ",\n    ".join(str(a) for a in annotations)
        text += "\n]"

        self.annotations_display.setPlainText(text)

    def save_annotations(self):
        if not self.annotations:
            self.statusBar.showMessage("No annotations to save")
            return

        default_filename = f"game_events_{self.current_game_id}.py" if self.current_game_id else "game_events.py"

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Annotations",
            default_filename,
            "Python files (*.py);;All files (*.*)"
        )

        if filename:
            with open(filename, 'w') as f:
                for game_id, events in self.annotations.items():
                    if events:
                        f.write(f'    "{game_id}": [\n')
                        f.write("        " + ", ".join(str(event) for event in events))
                        f.write("\n    ],\n")

            self.statusBar.showMessage(f"Saved annotations to {filename}")

def main():
    app = QApplication(sys.argv)
    window = AnnotationTool()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
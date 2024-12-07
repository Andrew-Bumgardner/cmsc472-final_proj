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

        # Add frame count QLabel
        self.frame_count = QLabel("Frame: 0 / 0")
        self.frame_count.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.frame_count)  # Add to the main layout

        # Range selection UI
        self.start_slider = QSlider(Qt.Horizontal)
        self.start_slider.setRange(0, 0)  # Updated after frames are loaded
        self.start_slider.valueChanged.connect(self.update_range_label)

        self.end_slider = QSlider(Qt.Horizontal)
        self.end_slider.setRange(0, 0)  # Updated after frames are loaded
        self.end_slider.valueChanged.connect(self.update_range_label)

        self.range_label = QLabel("Range: 0 - 0")

        layout.addWidget(QLabel("Start Frame:"))
        layout.addWidget(self.start_slider)
        layout.addWidget(QLabel("End Frame:"))
        layout.addWidget(self.end_slider)
        layout.addWidget(self.range_label)

        self.assign_range_btn = QPushButton("Assign Label to Range")
        self.assign_range_btn.clicked.connect(self.assign_label_to_range)
        layout.addWidget(self.assign_range_btn)

        layout.addWidget(frame_container)

        # Add frame slider with current frame label
        slider_layout = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.slider_changed)

        self.current_frame_label = QLabel("Current Frame: 0")
        self.current_frame_label.setAlignment(Qt.AlignRight)
        slider_layout.addWidget(self.frame_slider)
        slider_layout.addWidget(self.current_frame_label)

        layout.addLayout(slider_layout)

        # Event layout remains the same
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
            self.start_slider.setRange(0, len(self.frames)-1)
            self.end_slider.setRange(0, len(self.frames)-1)
            self.start_slider.setValue(0)
            self.end_slider.setValue(len(self.frames)-1)
            self.update_range_label()
            self.update_display()
            self.statusBar.showMessage(f"Loaded {len(self.frames)} frames from {self.current_game_id}")

    def update_range_label(self):
        start_frame = self.start_slider.value()
        end_frame = self.end_slider.value()
        if start_frame > end_frame:
            self.end_slider.setValue(start_frame)  # Ensure valid range
        self.range_label.setText(f"Range: {start_frame} - {end_frame}")
    
    def assign_label_to_range(self):
        start_frame = self.start_slider.value()
        end_frame = self.end_slider.value()
        event_id = self.event_group.checkedId()  # Get the selected event type
        if event_id == -1:
            self.statusBar.showMessage("No event type selected")
            return

        for frame_idx in range(start_frame, end_frame + 1):
            if self.current_game_id not in self.annotations:
                self.annotations[self.current_game_id] = []
            self.annotations[self.current_game_id] = [
                (frame, evt) for frame, evt in self.annotations[self.current_game_id]
                if frame < start_frame or frame > end_frame
            ]
            self.annotations[self.current_game_id].append((frame_idx, event_id))

        self.annotations[self.current_game_id].sort()
        self.update_annotations_display()
        self.statusBar.showMessage(f"Labeled frames {start_frame} to {end_frame} as {self.event_types[event_id]}")




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

        self.frame_count.setText(f"Frame: {self.frame_index+1} / {len(self.frames)}")  # Fixed
        self.frame_slider.setValue(self.frame_index)

        self.update_annotations_display()

    def jump_frames(self, delta):
        if not self.frames:
            return

        self.frame_index = max(0, min(self.frame_index + delta, len(self.frames) - 1))
        self.update_display()

    def slider_changed(self, value):
        self.frame_index = value
        self.current_frame_label.setText(f"Current Frame: {self.frame_index + 1}")
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
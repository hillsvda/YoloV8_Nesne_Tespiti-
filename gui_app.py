import sys
import os
import cv2
import numpy as np
import PyQt5
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins", "platforms")
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

from ultralytics import YOLO

import torch.serialization
import ultralytics.nn.tasks
import ultralytics.nn.modules.conv
import ultralytics.nn.modules.block
import torch.nn.modules.container

torch.serialization.add_safe_globals([
    ultralytics.nn.tasks.DetectionModel,
    torch.nn.modules.container.Sequential,
    ultralytics.nn.modules.conv.Conv,
    ultralytics.nn.modules.conv.RepConv,
    ultralytics.nn.modules.block.C2f,
    ultralytics.nn.modules.block.Bottleneck,
    ultralytics.nn.modules.block.SPPF,
    torch.nn.modules.conv.Conv2d,  # Conv2d güvenli listeye eklendi
    torch.nn.modules.batchnorm.BatchNorm2d,  # BatchNorm2d güvenli listeye eklendi
    torch.nn.modules.activation.SiLU,  # SiLU aktivasyon fonksiyonu güvenli listeye eklendi
    torch.nn.modules.container.ModuleList,  # ModuleList güvenli listeye eklendi
    torch.nn.modules.pooling.MaxPool2d,  # MaxPool2d güvenli listeye eklendi
    torch.nn.modules.upsampling.Upsample,  # Upsample güvenli listeye eklendi
    ultralytics.nn.modules.conv.Concat,  # Concat güvenli listeye eklendi
])
# --------------------------------------------------------------------

MODEL_PATH = r"C:\Users\huaweı\Desktop\Notebooks\runs\detect\yolov8_final_training\weights\best.pt"


class YoloDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 Nesne Tespiti - PyQt5")
        self.setGeometry(100, 100, 1100, 600)

        # MODEL YÜKLEME
        try:
            print("MODEL YÜKLENİYOR…")
            self.model = YOLO(MODEL_PATH)
            print("MODEL BAŞARIYLA YÜKLENDİ!")
        except Exception as e:
            print("HATA: Model yüklenemedi:", e)
            self.model = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        img_layout = QHBoxLayout()
        layout.addLayout(img_layout)

        self.orig_label = QLabel("Original Image")
        self.orig_label.setAlignment(Qt.AlignCenter)
        img_layout.addWidget(self.orig_label)

        self.tag_label = QLabel("Detected Image")
        self.tag_label.setAlignment(Qt.AlignCenter)
        img_layout.addWidget(self.tag_label)

        self.results_label = QLabel("Tespit Edilen Nesneler: 0")
        layout.addWidget(self.results_label)

        btns = QHBoxLayout()
        layout.addLayout(btns)

        self.btn_select = QPushButton("Select Image")
        self.btn_select.clicked.connect(self.select_img)

        self.btn_detect = QPushButton("Detect Objects")
        self.btn_detect.clicked.connect(self.detect)

        self.btn_save = QPushButton("Save Result")
        self.btn_save.clicked.connect(self.save_img)

        btns.addWidget(self.btn_select)
        btns.addWidget(self.btn_detect)
        btns.addWidget(self.btn_save)

        self.current_img = None
        self.detected_cv = None

    def select_img(self):
        f, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "", "Images (*.jpg *.png *.jpeg)")
        if f:
            self.current_img = f
            pix = QPixmap(f)
            if pix.isNull():
                # Try loading with OpenCV and convert to QImage
                img_cv = cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img_cv is not None:
                    rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                    h, w, _ = rgb.shape
                    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
                    self.orig_label.setPixmap(QPixmap.fromImage(qimg).scaled(
                        self.orig_label.width(), self.orig_label.height(),
                        Qt.KeepAspectRatio
                    ))
                    print(f"QPixmap OpenCV ile yüklendi: {f}")
                else:
                    self.orig_label.setText("Resim yüklenemedi! (Yol veya karakter sorunu olabilir)")
                    print(f"QPixmap ve OpenCV ile yüklenemedi: {f}")
                self.tag_label.clear()
            else:
                self.orig_label.setPixmap(pix.scaled(
                    self.orig_label.width(), self.orig_label.height(),
                    Qt.KeepAspectRatio
                ))
                self.tag_label.clear()

            self.results_label.setText("Tespit Edilen Nesneler: 0")


    def detect(self):
        if not self.current_img or not self.model:
            print("Resim yok veya model yüklenemedi.")
            return

        results = self.model(self.current_img)[0]

        img_arr = results.plot()
        self.detected_cv = img_arr

        rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)

        self.tag_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.tag_label.width(), self.tag_label.height(),
            Qt.KeepAspectRatio
        ))

        counts = {}
        for c in results.boxes.cls:
            name = self.model.names[int(c)]
            counts[name] = counts.get(name, 0) + 1

        total = sum(counts.values())
        txt = ", ".join([f"{n}: {c}" for n, c in counts.items()])

        self.results_label.setText(f"Tespit Edilen Nesneler ({total}): {txt}")

    def save_img(self):
        if self.detected_cv is None:
            print("Kaydedilecek görüntü yok.")
            return

        f, _ = QFileDialog.getSaveFileName(self, "Kaydet", "", "PNG (*.png);;JPG (*.jpg)")
        if f:
            cv2.imwrite(f, self.detected_cv)
            print("Kaydedildi:", f)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = YoloDetectorApp()
    win.show()
    sys.exit(app.exec())

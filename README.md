
# YOLOv8 ile Nesne Tespiti ve PyQt5 Arayüzü

Bu proje, kendi etiketlediğimiz özel bir veri seti (Cüzdan ve Mouse) üzerinde son teknoloji **YOLOv8** algoritmasını kullanarak nesne tespiti yapmak ve bu modeli **PyQt5** kütüphanesi ile tasarlanmış kullanıcı arayüzü üzerinden çalıştırmak amacıyla hazırlanmıştır.

##  İçerik

* **`proje2.ipynb`:** Roboflow'dan veri çekme, YOLOv8 modelinin (50 epoch) eğitimi ve başarı metriklerinin elde edildiği Jupyter Notebook dosyasıdır.
* **`gui_app.py`:** Eğitilen modeli yükleyerek görüntü seçme, nesne tespiti yapma ve sonucu kaydetme işlevlerini sağlayan PyQt5 arayüz uygulamasının Python kodudur.
* **`best.pt`:** 50 epoch eğitimi sonucunda elde edilen, en yüksek başarı skoruna sahip model ağırlık dosyasıdır.
* **`test_images/`:** Uygulamanın denenmesi için kullanılan örnek görüntüleri içerir.

## 🧠 Model Detayları ve Performansı

| Özellik | Değer |
| :--- | :--- |
| **Model Mimarisi** | YOLOv8 Nano (`yolov8n.pt`) |
| **Eğitim Süresi** | 50 Epoch |
| **Tespit Edilen Nesneler** | Mouse (Fare) ve Wallet (Cüzdan) |
| **Eğitim Başarısı (mAP50)** | **%98.7** |
| **Arayüz Teknolojisi** | PyQt5 |

## 💻 Uygulamanın Çalıştırılması

Uygulamayı yerel olarak çalıştırmak için aşağıdaki adımları izleyin:

1. **Gerekli Kütüphaneleri Yükleme:**
   ```bash
   pip install ultralytics PyQt5 opencv-python Pillow
   Uygulamayı Başlatma: Terminali açın, bu klasöre gidin ve gui_app.py dosyasını çalıştırın:
Kullanım:

Açılan pencerede Select Image butonuna basarak bir görüntü seçin.

Test Image butonuna basarak modelin nesne tespitini yapmasını sağlayın.

Save Image butonu ile tespit edilmiş (bounding box'lı) görüntüyü kaydedin.

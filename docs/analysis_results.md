# 😷 Analisis Project: Face Mask Detection with CNN

## 📌 Ringkasan Umum

Project ini adalah **Tugas Akhir (Skripsi)** dari mahasiswa:
- **Nama**: Rizky Amalia
- **NIM**: 19142037P
- **Prodi**: Teknik Informatika

Project bertujuan untuk **mendeteksi apakah seseorang menggunakan masker wajah atau tidak** menggunakan teknik **Deep Learning** — khususnya **Convolutional Neural Network (CNN)** dengan arsitektur **MobileNetV2** sebagai feature extractor melalui pendekatan **Transfer Learning**.

Selain deteksi, project ini juga melakukan **perbandingan performa 3 optimizer** yang berbeda: **Adam**, **SGD**, dan **RMSprop**.

> [!TIP]
> Project ini terkait jurnal ilmiah yang dipublikasikan di [Jurnal RESTI](http://jurnal.iaii.or.id/index.php/RESTI/article/view/4276).

---

## 📁 Struktur Folder Project

```
facemask-detection/
├── 0 Dataset/                          # Dataset gambar wajah
│   ├── masker/                         # 1000 gambar wajah DENGAN masker
│   │   ├── *-with-mask.jpg             # Gambar asli
│   │   ├── augmented_image_*.jpg       # Gambar hasil augmentasi
│   │   ├── k-mask_*.jpg                # Gambar tambahan (custom)
│   │   └── zNew_mask_1*.jpg            # Gambar tambahan (custom)
│   └── tidak_bermasker/                # 1029 gambar wajah TANPA masker
│       ├── *.jpg                       # Gambar asli
│       ├── augmented_image_*.jpg       # Gambar hasil augmentasi
│       ├── k-notmask_*.jpg             # Gambar tambahan (custom)
│       └── zzNo_mask*.jpg              # Gambar tambahan (custom)
│
├── 4 Face Detector/                    # Model detektor wajah (pre-trained)
│   ├── deploy.prototxt                 # Arsitektur jaringan Caffe
│   └── res10_300x300_ssd_iter_140000.caffemodel  # Weight model SSD (~10 MB)
│
├── Mod_Ev/                             # Hasil evaluasi model
│   ├── Ev_adam/                         # Grafik evaluasi optimizer Adam
│   │   └── loss_accuracy.png
│   ├── Ev_sgd/                          # Grafik evaluasi optimizer SGD
│   ├── Ev_rmsprop/                      # Grafik evaluasi optimizer RMSprop
│   ├── Gabungan/                        # Grafik perbandingan gabungan
│   ├── accuracy_plot.png                # Plot akurasi perbandingan
│   ├── loss_plot.png                    # Plot loss perbandingan
│   └── loss_accuracy.png                # Plot gabungan loss & accuracy
│
├── Mod_Save/                           # Model yang sudah di-training & disimpan
│   ├── model_adam/mobilenet_v2.model/   # Model hasil training dg Adam
│   ├── model_sgd/mobilenet_v2.model/    # Model hasil training dg SGD
│   ├── model_rmsprop/mobilenet_v2.model/ # Model hasil training dg RMSprop
│   ├── average_class report.png          # Laporan klasifikasi rata-rata
│   └── class report per optimizer.png    # Laporan klasifikasi per optimizer
│
├── training2.ipynb                      # 🔥 Notebook UTAMA: Training model
├── test video_adam.ipynb                 # Testing real-time dg model Adam
├── test video_sgd.ipynb                 # Testing real-time dg model SGD
├── test video_rmsprop.ipynb             # Testing real-time dg model RMSprop
├── Project Explanation.pptx             # Presentasi penjelasan project (~87 MB)
└── README.md                            # Dokumentasi project
```

---

## 🧠 Alur Kerja Project (Pipeline)

### 1. **Preprocessing Data**
- Load semua gambar dari folder `0 Dataset` (2029 gambar total)
- **Resize** gambar ke **224 × 224 piksel** (sesuai input MobileNetV2)
- **Konversi** gambar ke array NumPy
- **Scaling** nilai piksel dari `(0, 255)` ke `(-1, 1)` menggunakan `preprocess_input()`
- **Label Encoding**: Kategori (`masker` / `tidak_bermasker`) → biner → binary class matrix

### 2. **Data Splitting**
| Set         | Jumlah | Persentase |
|-------------|--------|------------|
| Training    | 1,379  | ~68%       |
| Validation  | 244    | ~12%       |
| Testing     | 406    | ~20%       |

- **Random State = 10** (agar hasil reprodusibel)
- **Stratify** digunakan untuk membagi data secara proporsional

### 3. **Data Augmentation**
Menggunakan `ImageDataGenerator` dengan parameter:
- Rotasi: ±20°
- Zoom: 15%
- Width/Height Shift: 20%
- Shear: 15%
- Horizontal & Vertical Flip
- Fill Mode: nearest

### 4. **Feature Extraction — Transfer Learning**
- Menggunakan **MobileNetV2** pre-trained pada **ImageNet** sebagai **feature extractor**
- Layer MobileNetV2 (base) di-**freeze** (non-trainable) → hanya head/classifier yang di-training
- Arsitektur head yang ditambahkan:

```
MobileNetV2 Output
    ↓
AveragePooling2D (7×7)
    ↓
Flatten
    ↓
Dense (128, relu)
    ↓
Dropout (0.5)
    ↓
Dense (2, softmax)  ← Output: [Masker, Tidak_Bermasker]
```

- **Total parameter**: 2,422,210
  - Trainable: 164,226 (hanya head classifier)
  - Non-trainable: 2,257,984 (MobileNetV2 yang di-freeze)

### 5. **Training — Perbandingan 3 Optimizer**
Model yang **sama** di-training dengan 3 optimizer berbeda:

| Optimizer | Keunggulan |
|-----------|------------|
| **Adam**    | Adaptive learning rate, paling populer |
| **SGD**     | Sederhana, stabil untuk dataset besar |
| **RMSprop** | Adaptive, baik untuk RNN & noisy gradient |

### 6. **Evaluasi Model**
- **Classification Report** (Precision, Recall, F1-Score)
- **Confusion Matrix**
- **Grafik Loss & Accuracy** per epoch untuk tiap optimizer
- **Perbandingan gabungan** antar optimizer

### 7. **Real-Time Face Mask Detection (Video)**
Notebook `test video_*.ipynb` melakukan:
1. Load **face detector** (SSD berbasis Caffe: `res10_300x300_ssd_iter_140000.caffemodel`)
2. Load **trained mask classifier** (model yang sudah disimpan)
3. Buka **webcam** (video stream real-time)
4. Untuk setiap frame:
   - Deteksi wajah menggunakan SSD
   - Crop & preprocess wajah yang terdeteksi
   - Prediksi: **Masker** atau **Tidak Bermasker**
   - Tampilkan bounding box + label + persentase confidence
   - 🟢 Hijau = Pakai masker | 🔴 Merah = Tidak pakai masker
5. Tekan `x` untuk keluar

---

## ⚙️ Yang Perlu Didownload & Diinstall

### 1. **Python** (versi 3.7.x)
Project ini dikembangkan menggunakan **Python 3.7.10** (terlihat dari metadata notebook). Disarankan versi **3.7.x** agar kompatibel.

> [!IMPORTANT]
> Versi Python sangat penting karena TensorFlow 2.x versi lama belum mendukung Python 3.8+ sepenuhnya.

### 2. **Conda** (Opsional tapi Direkomendasikan)
Project ini dikembangkan menggunakan **Miniconda** dengan virtual environment bernama `coba_env`.

```bash
# Install Miniconda
# Download dari: https://docs.conda.io/en/latest/miniconda.html

# Buat environment
conda create -n coba_env python=3.7
conda activate coba_env
```

### 3. **Library Python yang Harus Diinstall**

```bash
pip install tensorflow==2.4.0       # atau versi TF 2.x yang kompatibel dengan Python 3.7
pip install keras                    # (biasanya sudah bundled dengan TF 2.x)
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install opencv-python            # cv2 untuk deteksi wajah & video
pip install imutils                  # utility untuk video stream & image processing
pip install jupyter                  # atau jupyterlab untuk menjalankan notebook
```

**Ringkasan library:**

| Library | Kegunaan |
|---------|----------|
| `tensorflow` / `keras` | Framework deep learning, arsitektur MobileNetV2 |
| `numpy` | Operasi array/matrix |
| `pandas` | Manipulasi data tabular |
| `matplotlib` | Visualisasi grafik (loss, accuracy) |
| `seaborn` | Visualisasi statistik (confusion matrix, heatmap) |
| `scikit-learn` | Label encoding, data splitting, evaluasi model |
| `opencv-python` | Deteksi wajah (DNN module), video capture, image processing |
| `imutils` | Helper untuk video stream & resizing |
| `jupyter` | Menjalankan file `.ipynb` |

### 4. **Model Face Detector (Sudah Tersedia)**
File-file berikut **sudah ada di repo** (`4 Face Detector/`):
- `deploy.prototxt` — arsitektur jaringan SSD
- `res10_300x300_ssd_iter_140000.caffemodel` — weights model

### 5. **Webcam** (Untuk Testing Real-Time)
Notebook `test video_*.ipynb` membutuhkan **webcam aktif** (`VideoStream(src=0)`)

---

## ⚠️ Hal Penting Sebelum Menjalankan

> [!WARNING]
> **Path dataset di notebook masih hardcoded** ke lokasi asli pembuat:
> ```
> D:\OneDrive\Kuliah\KULIAH\MATA KULIAH\Semester 5\Project\0 Dataset
> ```
> Kamu **harus mengubah path ini** agar sesuai dengan lokasi folder project di komputermu. Hal yang sama berlaku untuk path model di notebook testing.

**Path yang perlu diubah:**

| File | Variabel | Path Asli |
|------|----------|-----------|
| `training2.ipynb` | `dataset` | Path ke `0 Dataset` |
| `test video_*.ipynb` | `prototxt` | Path ke `4 Face Detector/deploy.prototxt` |
| `test video_*.ipynb` | `weights` | Path ke `4 Face Detector/res10_300x300...caffemodel` |
| `test video_*.ipynb` | `maskNet` (load_model) | Path ke model yang tersimpan di `Mod_Save` |

---

## 🚀 Cara Menjalankan

### Training:
```bash
conda activate coba_env
jupyter notebook training2.ipynb
# Jalankan semua sel secara berurutan
```

### Testing Real-Time (contoh dg Adam):
```bash
jupyter notebook "test video_adam.ipynb"
# Pastikan webcam aktif
# Tekan 'x' untuk keluar dari video window
```

---

## 📊 Ringkasan

| Aspek | Detail |
|-------|--------|
| **Tujuan** | Deteksi masker wajah real-time |
| **Metode** | Transfer Learning + CNN (MobileNetV2) |
| **Dataset** | 2,029 gambar (1,000 masker + 1,029 tanpa masker) |
| **Input Size** | 224 × 224 × 3 (RGB) |
| **Optimizer Dibandingkan** | Adam, SGD, RMSprop |
| **Face Detector** | SSD (Single Shot Detector) berbasis Caffe |
| **Output** | 2 kelas: Masker / Tidak Bermasker |
| **Python Version** | 3.7.10 |
| **Framework** | TensorFlow / Keras |

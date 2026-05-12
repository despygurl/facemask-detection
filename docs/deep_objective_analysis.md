# 🔬 Deep Review & Analisis Objektif — Face Mask Detection Project

> Dokumen ini berisi analisis mendalam, temuan teknis, komentar objektif, serta saran perbaikan terhadap project Face Mask Detection menggunakan MobileNetV2.

---

## 1. Penilaian Umum

| Aspek | Skor | Keterangan |
|-------|------|------------|
| **Ide & Relevansi** | ⭐⭐⭐⭐⭐ | Sangat relevan di era pandemi COVID-19 |
| **Metodologi** | ⭐⭐⭐⭐ | Transfer Learning + perbandingan optimizer adalah pendekatan yang baik |
| **Kualitas Kode** | ⭐⭐⭐ | Fungsional tapi ada beberapa masalah struktural |
| **Dokumentasi** | ⭐⭐ | Minim — README sangat singkat, tidak ada `requirements.txt` |
| **Reprodusibilitas** | ⭐⭐ | Path hardcoded, tidak ada environment file |
| **Rigor Eksperimen** | ⭐⭐⭐ | Ada perbandingan optimizer, tapi SGD training gagal |

---

## 2. Analisis Arsitektur Model

### 2.1 Pemilihan MobileNetV2 — ✅ Tepat

Pemilihan **MobileNetV2** sebagai backbone adalah keputusan yang **sangat baik** untuk use case ini:

- **Lightweight**: Hanya ~3.4 juta parameter (cocok untuk deployment di edge device)
- **Efisien**: Menggunakan depthwise separable convolutions
- **Proven**: Performa tinggi di ImageNet dengan komputasi rendah
- Cocok untuk aplikasi real-time karena inference-nya cepat

### 2.2 Arsitektur Head Classifier — ⚠️ Bisa Dioptimalkan

```
MobileNetV2 (frozen) → AveragePooling2D(7×7) → Flatten → Dense(128, relu) → Dropout(0.5) → Dense(2, softmax)
```

**Komentar objektif:**

- Head classifier cukup sederhana — ini bisa menjadi **kelebihan** (menghindari overfitting) atau **kelemahan** (kurang ekspresif)
- Hanya ada **1 hidden layer (128 unit)** sebelum output — untuk binary classification ini sebenarnya sudah cukup
- **Dropout 0.5** adalah pilihan standar yang baik
- Tidak ada **Batch Normalization** di head — bisa ditambahkan untuk stabilitas training

> Total trainable params hanya **164,226** dari 2.4 juta total. Ini artinya 93% parameter di-freeze, yang merupakan pendekatan Transfer Learning yang konservatif tapi aman untuk dataset kecil.

### 2.3 Penggunaan Softmax untuk Binary — ⚠️ Redundan tapi Tidak Salah

Output layer menggunakan `Dense(2, softmax)` dengan `binary_crossentropy`. Secara teknis, untuk binary classification bisa lebih efisien menggunakan:

```python
Dense(1, activation='sigmoid')  # Lebih efisien untuk binary
```

Namun pendekatan saat ini (2 output + softmax) **tetap benar** dan menghasilkan output yang valid.

---

## 3. Analisis Dataset

### 3.1 Ukuran Dataset — ⚠️ Kecil tapi Cukup untuk Transfer Learning

| Kelas | Jumlah | Persentase |
|-------|--------|------------|
| `masker` | 1,000 | 49.3% |
| `tidak_bermasker` | 1,029 | 50.7% |
| **Total** | **2,029** | **100%** |

**Komentar:**

- Dataset **cukup seimbang** (hampir 50:50) — ini **bagus**, tidak perlu teknik oversampling/undersampling
- Ukuran 2,029 gambar adalah **relatif kecil** untuk deep learning, tapi karena menggunakan Transfer Learning, ini masih **acceptable**
- Dataset mengandung 3 sumber: gambar asli, augmented images, dan gambar tambahan (k-mask, zNew_mask) — menunjukkan upaya untuk memperbanyak variasi

### 3.2 Kualitas & Variasi Dataset — ⚠️ Perlu Diperhatikan

Berdasarkan inspeksi:

- Gambar memiliki **ukuran bervariasi** (5KB - 3MB) — ini menunjukkan resolusi asli yang sangat berbeda-beda
- Ada gambar **augmented** yang sudah ada di folder dataset — ini berarti **augmentasi dilakukan 2 kali** (sekali di file, sekali di training via `ImageDataGenerator`)
- Gambar `zNew_mask_1` dan `zzNo_mask 1` berukuran besar (~250-400KB) — kemungkinan foto resolusi tinggi dari sumber berbeda

> **⚠️ Data leakage potential**: Gambar augmented yang sudah ada di folder dataset (`augmented_image_*.jpg`) bisa saja masuk ke train DAN test set sekaligus, terutama jika augmentasi dari gambar yang sama. Ini bisa menghasilkan **metrik evaluasi yang terlalu optimis** (inflated).

### 3.3 Data Splitting — ✅ Baik tapi Ada Catatan

```python
# Split pertama: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(data, label2cat, test_size=0.2, random_state=10, stratify=label)

# Split kedua: 85% train, 15% validation (dari data training)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=10)
```

| Set | Jumlah | Persen Efektif |
|-----|--------|----------------|
| Training | 1,379 | 68% |
| Validation | 244 | 12% |
| Testing | 406 | 20% |

**Catatan:** Split kedua (validation) **tidak menggunakan stratify** — ini bisa menyebabkan distribusi kelas yang tidak merata di validation set. Meskipun dampaknya kecil untuk dataset yang sudah seimbang, ini tetap merupakan inkonsistensi metodologis.

---

## 4. Analisis Training

### 4.1 Hyperparameter

| Parameter | Nilai | Komentar |
|-----------|-------|----------|
| Learning Rate | `1e-3` (0.001) | Agak tinggi untuk fine-tuning, biasanya `1e-4` lebih umum |
| Epochs | 50 | Cukup, tapi tanpa EarlyStopping |
| Batch Size | 20 | Tidak standar (biasanya 16 atau 32), tapi tidak masalah |
| LR Decay | `lr/epochs` = 0.00002 per epoch | Decay rate sangat kecil |
| Loss Function | `binary_crossentropy` | Tepat untuk binary classification |

> Learning rate **1e-3** cukup tinggi untuk Transfer Learning. Umumnya, `1e-4` atau `1e-5` lebih disarankan agar weight pre-trained tidak rusak. Namun karena base model di-freeze seluruhnya, ini tidak terlalu menjadi masalah — hanya head classifier yang di-training.

### 4.2 Data Augmentation — ✅ Baik

```python
ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
```

**Komentar:**

- Parameter augmentasi sudah **cukup agresif** — bagus untuk mencegah overfitting
- **`vertical_flip=True`** untuk wajah mungkin **kurang natural** — wajah manusia jarang terlihat terbalik dalam konteks real-world. Namun dampaknya kecil karena model belajar fitur lokal

### 4.3 Hasil Training per Optimizer

#### Adam — ✅ Berhasil Sempurna (50 epoch)

| Metrik | Epoch 1 | Epoch 25 | Epoch 50 |
|--------|---------|----------|----------|
| Train Accuracy | 78.96% | 93.08% | 95.07% |
| Val Accuracy | 91.67% | 93.75% | 94.58% |
| Train Loss | 0.4609 | 0.1631 | 0.1204 |
| Val Loss | 0.2292 | 0.1294 | 0.1242 |

**Analisis:**

- Konvergensi yang **baik dan stabil**
- Gap antara train dan val accuracy kecil (~0.5%) → **tidak overfitting**
- Val accuracy terbaik: **95.42%** (epoch 42)
- Val loss fluktuatif tapi trendnya menurun — menunjukkan model masih bisa belajar

#### SGD — ❌ Training Dibatalkan (KeyboardInterrupt di Epoch 12)

| Metrik | Epoch 1 | Epoch 11 (terakhir) |
|--------|---------|---------------------|
| Train Accuracy | 56.59% | 78.51% |
| Val Accuracy | 59.58% | 77.50% |
| Train Loss | 0.7906 | 0.4467 |
| Val Loss | 0.6669 | 0.4469 |

**Analisis:**

- SGD **diinterupsi secara manual** (KeyboardInterrupt) di awal epoch 12
- Konvergensi SGD memang **jauh lebih lambat** dibanding Adam — ini expected karena SGD tidak memiliki adaptive learning rate
- Pada saat dibatalkan, akurasi baru **~78%** vs Adam yang sudah **~89%** di epoch serupa
- **Model SGD yang tersimpan hanya ditraining 11 epoch** — ini membuat perbandingan dengan Adam (50 epoch) menjadi **tidak adil/tidak valid**

> **⚠️ Temuan Kritis**: Training SGD **tidak selesai** — dibatalkan paksa oleh user. Ini berarti perbandingan performa SGD vs Adam vs RMSprop dalam jurnal/skripsi **tidak apple-to-apple**. SGD seharusnya juga dijalankan 50 epoch penuh agar perbandingan valid.

#### RMSprop — ⚠️ Perlu Diperiksa Lebih Lanjut

Dari notebook, terlihat kode compile RMSprop ada tapi training output perlu diperiksa lebih lanjut karena file notebook sangat besar. Berdasarkan adanya folder `Mod_Save/model_rmsprop/` yang berisi model, training RMSprop kemungkinan berhasil.

### 4.4 Tidak Ada Early Stopping — ⚠️ Masalah

Kode early stopping dan custom callback ada di notebook **tapi di-comment out**:

```python
# my_callbacks = [
#     tf.keras.callbacks.EarlyStopping(patience=7, monitor='val_accuracy', mode='max', restore_best_weights=True),
#     tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', mode='min', restore_best_weights=True)
# ]
```

**Dampak:**

- Model yang disimpan adalah model di **epoch terakhir** (epoch 50), bukan model terbaik
- Untuk Adam, val accuracy di epoch 42 (95.42%) lebih baik dari epoch 50 (94.58%)
- Tanpa `restore_best_weights`, model akhir **bukan yang terbaik**

---

## 5. Analisis Kode — Bug & Issues

### 5.1 🐛 Bug Kritis di `detect_and_predict_mask()`

```python
for i in range(0, detections.shape[2]):
    confidence = detections[0,0,i,2]
    if confidence > 0.5:
        # ... proses wajah ...
        wajah.append(face)
        lokasi.append(...)

    # BUG: if/return ini ADA DI DALAM LOOP!
    if len(wajah) > 0:
        wajah = np.array(wajah, dtype='float32')
        preds = maskNet.predict(wajah, batch_size=12)
    
    return(lokasi, preds)  # ← RETURN DI DALAM LOOP!
```

**Masalah:** Blok `if len(wajah) > 0` dan `return` berada **di dalam for loop** (perhatikan indentasi). Ini berarti:

- Fungsi **selalu return setelah iterasi pertama** loop
- Jika deteksi wajah pertama confidence < 0.5, fungsi return dengan `wajah=[]` dan `preds=[]`
- Jika deteksi wajah pertama confidence > 0.5, hanya **1 wajah** yang terdeteksi (meskipun ada banyak wajah)

**Fix yang seharusnya:**

```python
for i in range(0, detections.shape[2]):
    confidence = detections[0,0,i,2]
    if confidence > 0.5:
        # ... proses wajah ...

# Seharusnya di LUAR loop:
if len(wajah) > 0:
    wajah = np.array(wajah, dtype='float32')
    preds = maskNet.predict(wajah, batch_size=12)

return(lokasi, preds)
```

> **⚠️ Bug ini membuat deteksi hanya bekerja untuk 1 wajah** (atau gagal jika wajah pertama yang di-scan memiliki confidence rendah). Ini bukan masalah kecil — ini membuat fungsi inti deteksi **tidak reliable** untuk skenario multi-face.

### 5.2 Path Hardcoded — ⚠️ Tidak Portable

Semua path di-hardcode ke lokasi spesifik:

```python
dataset = r'D:\OneDrive\Kuliah\KULIAH\MATA KULIAH\Semester 5\Project\0 Dataset'
```

**Seharusnya menggunakan relative path:**

```python
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset = os.path.join(BASE_DIR, '0 Dataset')
```

### 5.3 Kode Duplikasi — ⚠️ Poor Practice

3 notebook testing (`test video_adam.ipynb`, `test video_sgd.ipynb`, `test video_rmsprop.ipynb`) memiliki kode yang **100% identik** kecuali path model. Seharusnya menggunakan **1 notebook/script dengan parameter**.

### 5.4 Tidak Ada `requirements.txt` — ❌

Tidak ada file yang mendokumentasikan dependencies. Orang lain yang mau menjalankan project harus menebak-nebak versi library.

---

## 6. Analisis Kelengkapan Eksperimen

### 6.1 Apa yang Sudah Baik ✅

1. **Perbandingan 3 optimizer** — memberikan perspektif yang lebih kaya
2. **Augmentasi data** — menghindari overfitting pada dataset kecil
3. **Transfer Learning** — pendekatan tepat untuk dataset terbatas
4. **Real-time testing** — membuktikan model bisa bekerja di dunia nyata
5. **Evaluasi multi-metrik** — ada accuracy, loss, classification report, confusion matrix
6. **Grafik perbandingan** — visualisasi training per optimizer

### 6.2 Apa yang Kurang ❌

| Aspek | Komentar |
|-------|----------|
| **Cross-validation** | Tidak dilakukan — hanya 1 kali split. K-fold cross-validation akan memberikan estimasi performa yang lebih robust |
| **Reproducibility** | Tidak semua random seed dikontrol (augmentasi, weight init) |
| **Ablation study** | Tidak ada eksperimen untuk memahami kontribusi masing-masing komponen |
| **Confusion matrix detail** | Ada file PNG tapi tidak ada analisis per-kelas di notebook |
| **Testing dataset independen** | Tidak ada pengujian dengan dataset dari sumber yang benar-benar berbeda |
| **Latency benchmarking** | Tidak ada pengukuran FPS/inference time — padahal ini penting untuk real-time |
| **SGD training tidak selesai** | Perbandingan menjadi tidak valid |
| **Fine-tuning beberapa layer** | Tidak ada eksperimen unfreeze beberapa layer terakhir MobileNetV2 |

---

## 7. Analisis Kode Quality

### 7.1 Kelebihan

- Komentar dalam Bahasa Indonesia yang cukup deskriptif di notebook
- Setiap section notebook diberi heading markdown yang jelas
- Variabel memiliki nama yang bermakna (`wajah`, `lokasi`, `preds`)
- Preprocessing pipeline yang benar sesuai rekomendasi MobileNetV2

### 7.2 Kekurangan

- Tidak ada **error handling** di seluruh kode
- Tidak ada **logging** selama training/testing
- Tidak ada **unit test**
- Kode augmentasi contoh di-comment out — menunjukkan eksperimen yang tidak rapi
- Banyak kode yang di-comment out dibiarkan di notebook final — sebaiknya dibersihkan
- Tidak ada **modularisasi** — semua kode ada di notebook, bukan Python module

---

## 8. Saran Perbaikan (Prioritas Tinggi ke Rendah)

### 🔴 Prioritas Tinggi

1. **Fix bug indentasi di `detect_and_predict_mask()`** — ini bug yang membuat multi-face detection tidak berfungsi
2. **Jalankan ulang training SGD sampai 50 epoch** — agar perbandingan optimizer valid
3. **Tambahkan EarlyStopping + ModelCheckpoint** — simpan model terbaik, bukan model terakhir
4. **Buat `requirements.txt`** — agar project bisa di-reproduce oleh orang lain

### 🟡 Prioritas Sedang

1. **Gunakan relative path** — ganti semua hardcoded path
2. **Gabungkan 3 notebook testing** jadi 1 script dengan argumen
3. **Periksa data leakage** — pastikan augmented images di dataset tidak bocor antara train/test
4. **Tambahkan validation stratify** di split kedua
5. **Tambahkan FPS counter** di real-time testing

### 🟢 Prioritas Rendah

1. **Coba fine-tune beberapa layer terakhir** MobileNetV2 (unfreeze 20-30 layer terakhir)
2. **Tambahkan kelas ketiga**: "masker tidak benar" (chin mask, nose out)
3. **Eksperimen dengan learning rate scheduler** (ReduceLROnPlateau)
4. **Bersihkan commented-out code** dari notebook final
5. **Tambahkan K-Fold Cross Validation** untuk evaluasi yang lebih robust

---

## 9. Perbandingan dengan State-of-the-Art

| Aspek | Project Ini | Best Practice |
|-------|------------|---------------|
| Backbone | MobileNetV2 | ✅ Masih relevan, tapi EfficientNet/ConvNeXt lebih modern |
| Dataset Size | 2,029 | ⚠️ Dataset publik seperti RMFD memiliki 90,000+ gambar |
| Classes | 2 (binary) | ⚠️ Dataset modern memiliki 3+ kelas (correct mask, incorrect, no mask) |
| Detection | SSD (Caffe) | ⚠️ YOLOv5/v8 atau MediaPipe lebih cepat dan akurat |
| Framework | TF 2.x / Keras | ✅ OK, tapi PyTorch kini lebih populer di research |
| Deployment | Jupyter Notebook | ❌ Seharusnya ada script standalone / Flask API / mobile deployment |

---

## 10. Kesimpulan Objektif

### Yang Sudah Baik

Project ini menunjukkan **pemahaman fundamental yang baik** tentang deep learning, transfer learning, dan computer vision. Pemilihan MobileNetV2 tepat untuk use case ringan, dan upaya membandingkan 3 optimizer menunjukkan mindset eksperimental yang baik. Untuk level Tugas Akhir/Skripsi, project ini **memenuhi standar minimum** yang diharapkan.

### Yang Perlu Diperbaiki

Namun, project ini memiliki beberapa **kelemahan metodologis yang signifikan**:

1. SGD training yang tidak selesai membuat **klaim perbandingan optimizer tidak valid**
2. Bug di fungsi deteksi membuat **multi-face detection gagal**
3. Kurangnya reproducibility infrastructure (requirements.txt, relative paths, environment files)
4. Tidak ada early stopping, sehingga model yang disimpan **bukan model terbaik**

### Rating Keseluruhan

| Kriteria | Rating |
|----------|--------|
| Untuk Tugas Akhir S1 | ⭐⭐⭐⭐ — **Baik**, dengan catatan perbaikan di atas |
| Untuk Production Use | ⭐⭐ — **Belum siap**, perlu banyak perbaikan |
| Untuk Research Paper | ⭐⭐⭐ — **Cukup**, tapi perlu rigor yang lebih ketat |

---

*Dokumen ini dibuat sebagai review objektif untuk keperluan analisis teknis. Semua temuan berdasarkan inspeksi kode dan output yang ada di repository.*

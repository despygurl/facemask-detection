# facemask-detection

Artikel ini membahas tentang pentingnya menggunakan masker untuk memutus penyebaran COVID-19 dan bagaimana teknologi Convolutional Neural Network (CNN) dapat digunakan untuk mendeteksi penggunaan masker yang tepat. CNN merupakan algoritma Deep Learning yang populer untuk masalah klasifikasi data gambar. Penelitian ini menggunakan model MobileNetV2 yang telah dilatih sebelumnya untuk membantu membangun Mask Usage Detector. Selain itu, penelitian ini membandingkan performa tiga metode optimisasi dari CNN, yaitu Adam, SGD, dan RMSprop dalam mendeteksi penggunaan masker. Performa diukur berdasarkan hasil uji coba dengan menganalisis nilai akurasi, presisi, dan recall. Dataset yang digunakan adalah gambar berjumlah 2.029, terdiri dari kategori "dengan masker" dan "tanpa masker". Hasil uji coba menunjukkan bahwa optimisasi Adam memiliki performa terbaik dengan akurasi mencapai 93,84%. Dengan demikian, penelitian ini memberikan kontribusi penting dalam mengembangkan teknologi deteksi penggunaan masker yang efektif untuk membantu memutus penyebaran COVID-19.

#

This article discusses the importance of using masks to prevent the spread of COVID-19 and how Convolutional Neural Network (CNN) technology can be used to detect proper mask usage. CNN is a popular Deep Learning algorithm for image data classification problems. This research uses the pre-trained MobileNetV2 model to help build the Mask Usage Detector. In addition, this study compares the performance of three optimization methods of CNN, namely Adam, SGD, and RMSprop, in detecting mask usage. The performance is measured based on the results of testing by analyzing the accuracy, precision, and recall values. The dataset used consists of 2,029 images, consisting of categories of "with mask" and "without mask". The test results show that the Adam optimization has the best performance with an accuracy of up to 93.84%. Therefore, this research provides an important contribution to the development of effective mask usage detection technology to help break the chain of COVID-19 transmission.

### Publication/Article link: [Article](http://jurnal.iaii.or.id/index.php/RESTI/article/view/4276)
### PPT Project: [Project Explanation.pptx](https://drive.google.com/file/d/1NzFcBGYx8m8SGTooJa2dBlq7K-fYU8v1/view?usp=sharing)

# 😷 Face Mask Detection with CNN

A deep learning project that classifies whether a person is wearing a face mask, using CNN (MobileNetV2) and comparing three optimizers: Adam, SGD, and RMSprop.

## 📌 Overview
This project was developed as part of an academic study and aims to:
- Help detect proper mask usage using image classification
- Compare optimizer performance (Adam, SGD, RMSprop)
- Achieve a robust model using MobileNetV2 as backbone

Dataset: 2029 labeled face images  
Classes: `with_mask`, `without_mask`

## 🔧 How to Run

1. Clone this repository:

<pre><code>1. Clone this repository: ```bash git clone https://github.com/despygurl/facemask-detection.git cd facemask-detection ``` </code></pre>

## 🔗 Article & Presentation

- [Jurnal RESTI](http://jurnal.iaii.or.id/index.php/RESTI/article/view/4276)
- [PPT Project](https://drive.google.com/file/d/1NzFcBGYx8m8SGTooJa2dBlq7K-fYU8v1/view?usp=sharing)

# Laporan Proyek Machine Learning - Harry Mardika

## Domain Proyek

PT Bank Central Asia Tbk (BBCA.JK), yang lebih dikenal sebagai Bank Central Asia (BCA), adalah salah satu institusi keuangan terkemuka di Indonesia. Berdiri sejak tahun 1955, BCA telah menjadi pilar dalam industri perbankan dengan menyediakan beragam layanan perbankan untuk konsumen, korporasi, investasi, dan manajemen aset. Dengan jaringan yang luas meliputi cabang dan ATM di seluruh Indonesia, BCA diakui atas kinerja keuangannya yang kuat, inovasi dalam layanan perbankan, dan komitmen terhadap kepuasan pelanggan.

Sebagai salah satu dari bank terbesar di Indonesia yang terdaftar di bursa saham, pergerakan harga saham BCA memiliki dampak yang signifikan bagi investor dan perencanaan strategis perusahaan. Dalam upaya untuk membantu mengantisipasi perubahan harga saham di masa depan, teknik Machine Learning, khususnya model Long Short Term Memory (LSTM), dapat digunakan. LSTM, sebagai bentuk dari Recurrent Neural Network, dirancang khusus untuk menganalisis dan memahami pola-pola kompleks dalam data deret waktu seperti harga saham.

Dengan memanfaatkan data historis, prediksi yang akurat mengenai pergerakan harga saham BCA tidak hanya memberikan pandangan yang lebih baik bagi investor untuk mengambil keputusan yang tepat, tetapi juga membantu perusahaan dalam merumuskan strategi bisnis yang lebih efektif. Dengan demikian, pendekatan ini dapat berperan penting dalam mengurangi risiko kerugian finansial dan meningkatkan kinerja investasi secara keseluruhan.

## Business Understanding

### Problem Statements

- Prediksi harga saham adalah tugas yang sangat kompleks dan selalu ada risiko. Bagaimana kita dapat meminimalkan risiko ini?
- Bagaimana kita dapat menggunakan data historis untuk memprediksi pergerakan harga saham di masa depan?
  Menjelaskan pernyataan masalah latar belakang:

### Goals

Goals

- Mengembangkan model Machine Learning yang dapat memprediksi harga saham dengan akurasi yang tinggi, sehingga dapat membantu investor membuat keputusan investasi yang lebih baik dan mengurangi risiko kerugian finansial.
- Memprediksi harga saham untuk 30 hari kedepan per tanggal 20 Februari 2024

### Solution statements

Solution Statements

- Membuat model Long Short Term Memory (LSTM) dengan menggunakan library tensorflow dan pytorch untuk memprediksi harga saham. LSTM adalah jenis Recurrent Neural Network yang dirancang untuk belajar dependensi jangka panjang, yang sangat berguna untuk memprediksi data deret waktu seperti harga saham.
- Melakukan hyperparameter tuning pada model LSTM dengan menggunakan library kerastuner dan optuna untuk meningkatkan akurasi prediksi. Metrik evaluasi yang akan kita gunakan adalah Mean Squared Error (MSE) dan Root Mean Squared Error (RMSE)

## Data Understanding

Dataset ini berisi data harga saham historis PT Bank Central Asia Tbk (BBCA. JK) dari tahun 8 Juni 2004 hingga 20 Februari 2024. Dataset ini mencakup harga saham harian, volume perdagangan, dan metrik keuangan relevan lainnya untuk bank-bank terkemuka. Harga saham disediakan dalam mata uang IDR (Rupiah Indonesia).

Referensi : [Yahoo finance Saham BCA](https://finance.yahoo.com/quote/BBCA.JK/history/)

### Fitur-fitur yang terdapat pada dataset BBCA.JK:

- Date: Tanggal data harga saham.
- Open: Harga pembukaan saham bank pada tanggal tertentu.
- Close: Harga penutupan saham bank pada tanggal tertentu.
- High: Harga tertinggi yang dicapai oleh saham bank selama hari perdagangan.
- Low: Harga terendah yang dicapai oleh saham bank selama hari perdagangan.
- Adj Close: Harga penutupan pada hari perdagangan tertentu, disesuaikan untuk mencerminkan tindakan korporasi, seperti pemecahan saham, dividen, penawaran hak, atau penyesuaian lain yang dapat mempengaruhi harga saham.
- Volume: Jumlah saham yang diperdagangkan pada tanggal tertentu.
  Sumber Data:
  Himpunan data disusun dari sumber keuangan yang andal, termasuk bursa saham, situs web berita keuangan, dan penyedia data keuangan terkemuka. Teknik pembersihan dan preprocessing data telah diterapkan untuk memastikan akurasi dan konsistensi.

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:

### Data Analysis

- Ringkasan statistik deskriptif dari Dataset


**Rubrik/Kriteria Tambahan (Opsional)**:

- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation

Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling

Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation

Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:

- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

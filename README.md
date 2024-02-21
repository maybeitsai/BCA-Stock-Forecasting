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

- Mengembangkan model Machine Learning yang dapat memprediksi harga saham dengan akurasi yang tinggi, sehingga dapat membantu investor membuat keputusan investasi yang lebih baik dan mengurangi risiko kerugian finansial.
- Memprediksi harga saham untuk 30 hari kedepan per tanggal 20 Februari 2024

### Solution statements

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
  
  count : Jumlah data
  
  mean : nilai rata-rata
  
  min : nilai data minimum
  
  25% : kuartil pertama (Q1)
  
  50% : kuartil kedua (Median)
  
  75% : kuartil ketiga (Q3)
  
  max : nilai data maximum
  
  std : standar deviasi
![Describe](https://github.com/maybeitsai/BCA-Stock-Forecasting/assets/130530985/e76e20cf-786a-402c-831c-7b5fe2346a37)

- Mengamati hubungan antar fitur numerik dengan fungsi pairplot()

![cor plot](https://github.com/maybeitsai/BCA-Stock-Forecasting/assets/130530985/e8cb1b43-b028-43c1-8c2f-2e700894d746)

- Membuat heatmap korelasi antar fitur

Berdasarkan diagram heatmap, banyak fitur yang memiliki korelasi tinggi, sedangkan volume memiliki korelasi negatif

![corr](https://github.com/maybeitsai/BCA-Stock-Forecasting/assets/130530985/3e723bb3-3d78-4555-b18b-750a2eb3ef41)

- Menganalisa fitur yang memiliki korelasi tinggi

Setelah dianalisa ternyata fitur-fitur tersebut memiliki nilai yang tidak jauh berbeda

![Price](https://github.com/maybeitsai/BCA-Stock-Forecasting/assets/130530985/2520873c-8310-49ac-a009-081af9c1f14b)

- Menganalisa Moving Average dan Close Price
  
  Moving Average digunakan untuk memperhalus data harga penutupan saham selama 50 dan 200 hari terakhir. Dengan memvisualisasikan kedua moving average ini bersama harga penutupan sebenarnya, Anda dapat melihat tren jangka pendek dan jangka panjang dalam harga saham BCA.

![Close](https://github.com/maybeitsai/BCA-Stock-Forecasting/assets/130530985/e7784995-8ad6-495d-8d38-f2922f20ab2a)

- Mengamati volume perdagangan dan Moving Average

  ![Volume](https://github.com/maybeitsai/BCA-Stock-Forecasting/assets/130530985/ed18fc9f-f38f-4665-9092-87df187868f5)




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

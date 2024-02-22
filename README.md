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
- Memprediksi harga saham untuk 30 hari kedepan dari tanggal 21 Februari 2024

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

### Eksplorasi Data Analysis (EDA)

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

  Bentuk distribusi ini dapat memberikan informassi tentang hubungan setiap variabel. Pola dalam plot ini dapat menunjukkan korelasi antara variabel.

![cor plot](https://github.com/maybeitsai/BCA-Stock-Forecasting/assets/130530985/e8cb1b43-b028-43c1-8c2f-2e700894d746)

- Membuat heatmap korelasi antar fitur

Berdasarkan diagram heatmap, banyak fitur yang memiliki korelasi tinggi, sedangkan volume memiliki korelasi negatif

![corr](https://github.com/maybeitsai/BCA-Stock-Forecasting/assets/130530985/3e723bb3-3d78-4555-b18b-750a2eb3ef41)

- Mengamati volume perdagangan dan Moving Average

  Menghitung moving average (MA) dari volume perdagangan saham dengan menggunakan jendela (window) 20 hari terakhir. MA volume perdagangan membantu dalam memahami tren atau pola pergerakan volume perdagangan saham selama periode waktu tertentu.

  ![Volume](https://github.com/maybeitsai/BCA-Stock-Forecasting/assets/130530985/ed18fc9f-f38f-4665-9092-87df187868f5)

- Menganalisa fitur yang memiliki korelasi tinggi

Setelah dianalisa ternyata fitur-fitur tersebut memiliki nilai yang tidak jauh berbeda

![Price](https://github.com/maybeitsai/BCA-Stock-Forecasting/assets/130530985/2520873c-8310-49ac-a009-081af9c1f14b)

- Menganalisa Moving Average dan Close Price
  
  Moving Average digunakan untuk memperhalus data harga penutupan saham selama 50 dan 200 hari terakhir. Dengan memvisualisasikan kedua moving average ini bersama harga penutupan sebenarnya, Kita dapat melihat tren jangka pendek dan jangka panjang dalam harga saham BCA.

![Close](https://github.com/maybeitsai/BCA-Stock-Forecasting/assets/130530985/e7784995-8ad6-495d-8d38-f2922f20ab2a)

## Data Preparation

### Seleksi fitur

Mengambil dua kolom, yaitu kolom "Date" dan "Close", dari DataFrame yang disebut df_analysis. Kolom "Close" ini akan digunakan sebagai kolom target dan diubah menjadi bentuk array satu dimensi menggunakan metode .values.reshape(-1, 1). Hal ini dilakukan untuk memastikan data siap digunakan dalam proses selanjutnya.

### Pembagian dataset 

Membagi dataset menjadi dua bagian yaitu data training dan data testing dengan menggunakan fungsi train_test_split dari library sklearn. Data training digunakan untuk melatih model, sedangkan data testing digunakan untuk menguji kinerja model pada data yang belum pernah dilihat sebelumnya. Dalam contoh ini, data dipisahkan menggunakan metode train_test_split dengan ukuran data testing sebesar 20% dari total data (test_size=0.2). Penyebutan shuffle=False menunjukkan bahwa data tidak diacak sebelum dipisahkan, yang berarti urutannya dipertahankan. Hal ini penting terutama jika data yang Anda miliki memiliki sifat kronologis atau urutan tertentu yang harus dipertahankan dalam pembagian data training dan testing. Adapun jumlah data training sebanyak 3904 dan jumlah data testing sebanyak 977.

### Normalisasi

Normalisasi data merupakan proses untuk mengubah rentang nilai dari setiap fitur dalam dataset menjadi rentang yang seragam. Dalam proses ini, saya menggunakan MinMaxScaler dari library scikit-learn untuk melakukan normalisasi. MinMaxScaler akan mengubah nilai-nilai data sehingga berada dalam rentang antara 0 dan 1.

### Mengubah dimensi dan menentukan variabel x dan y

Pada tahap ini saya membuat fungi prepare_data yang berguna untuk memisahkan variabel x. Setiap 100 data digunakan untuk memprediksi data berikutnya yang merupakan variabel y. Lalu pada fungsi prepare_data saya juga mengubah fitur (X) dan target (y) menjadi array numpy untuk kemudian digunakan dalam model. Data fitur diubah menjadi format tiga dimensi yang diperlukan oleh model LSTM.

## Modeling

Pada tahap ini, saya memilih menggunakan model LSTM (Long Short-Term Memory). Penggunaan model LSTM dalam tahap pemodelan dipilih karena keunggulan-keunggulan tertentu yang dimilikinya, terutama dalam penanganan data deret waktu seperti data keuangan, cuaca, atau bahkan teks. Saya membuat dua model LSTM dengan menggunakan dua library yang berbeda yaitu Tensorflow dan Pytorch. Serta menggunakan library Kerastuner dan Optuna untuk melakukan penyetelan parameternya.

### LSTM dengan library Tensorflow dan Kerastuner

Model ini merupakan jenis sekuensial yang sering digunakan pada jaringan saraf tiruan (neural networks). Terdapat dua lapisan LSTM yang diikuti oleh beberapa lapisan Dense. Setiap lapisan memiliki berbagai parameter yang dapat disesuaikan untuk penyesuaian model, seperti jumlah unit (units), fungsi aktivasi, dll. 

Pada lapisan LSTM pertama dan kedua saya melakukan hyperparameter tuning dengan jumlah unit antara 16-256 yang memiliki 16 langkah (step). Pada lapisan LSTM pertama merupakan lapisan yang menerima inputan. Pada Lapisan ini terdapat return_sequences=True yang berguna untuk mengembalikan urutan output yang lengkap dari setiap time step. Hal ini berguna ketika layer LSTM tersebut diikuti oleh layer LSTM atau layer lain yang memerlukan urutan output dari setiap time step.

Setelah dua lapisan LSTM, model menggunakan beberapa lapisan Dense untuk melakukan pemrosesan fitur yang dihasilkan oleh lapisan LSTM sebelumnya. Dalam model yang ini, terdapat tiga lapisan Dense. Lapisan Dense pertama menggunakan parameter units yang diatur dengan fungsi hyperparameter dengan jumlah unit antara 16-128 yang memiliki 16 langkah (step). Lapisan Dense kedua menggunakan parameter units yang diatur dengan fungsi hyperparameter dengan jumlah unit antara 8-64 yang memiliki 8 langkah (step). lapisan Dense terakhir memiliki satu neuron, yang bertanggung jawab untuk menghasilkan output akhir dari model yaitu, prediksi yang diinginkan. 

Setelah parameter yang ingin diatur ditetapkan, selanjutnya model dicompile dengan menggunakan optimer Adam dengan learning rate yang disetel juga dengan nilai 0.01, 0.001, dan 0.0001

Pada model ini juga menggunakan beberapa fungsi Callback yaitu :
- EarlyStopping : berfungsi untuk menghentikan pelatihan lebih awal jika metrik yang dipantau tidak meningkat setelah sejumlah epoch tertentu (patience) dan mengembalikan bobot model ke iterasi terbaik selama pelatihan.
  
- ModelCheckPoint : berfungsi untuk menyimpan model ke dalam file best_model.h5 hanya jika nilai metrik yang dipanta terbaik dari semua epoch yang telah dilalui dan memastikan bahwa hanya model dengan performa terbaik yang disimpan.

- ReduceLROnPlateau : Mengurangi laju pembelajaran (learning rate) jika tidak ada peningkatan dalam metrik yang dipantau setelah sejumlah epoch tertentu (patience)

- rmse_threshold_callback : bertujuan untuk menghentikan proses pelatihan model jika nilai RMSE (Root Mean Squared Error) pada data latih dan validasi sudah mencapai batas tertentu.

#### Best Model

Setelah dilakukannya hyperparameter tuning, model dipilih dengan parameter yang terbaik. Adapun parameter terbaik pada model ini sebagai berikut.
  
![image](https://github.com/maybeitsai/BCA-Stock-Forecasting/assets/130530985/b1a5a77d-225b-4933-9b22-85ea9fc72bd0)



### LSTM dengan library Pytorch dan Optuna

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

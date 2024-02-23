# Laporan Proyek Machine Learning - Harry Mardika

## Domain Proyek

PT Bank Central Asia Tbk (BBCA.JK), yang lebih dikenal sebagai Bank Central Asia (BCA), adalah salah satu institusi keuangan terkemuka di Indonesia. Berdiri sejak tahun 1955, BCA telah menjadi pilar dalam industri perbankan dengan menyediakan beragam layanan perbankan untuk konsumen, korporasi, investasi, dan manajemen aset. Dengan jaringan yang luas meliputi cabang dan ATM di seluruh Indonesia, BCA diakui atas kinerja keuangannya yang kuat, inovasi dalam layanan perbankan, dan komitmen terhadap kepuasan pelanggan.

Sebagai salah satu dari bank terbesar di Indonesia yang terdaftar di bursa saham, pergerakan harga saham BCA memiliki dampak yang signifikan bagi investor dan perencanaan strategis perusahaan. Dalam upaya untuk membantu mengantisipasi perubahan harga saham di masa depan, teknik _Machine Learning_, khususnya model _Long Short-Term Memory (LSTM)_, dapat digunakan. _LSTM_, sebagai bentuk dari _Recurrent Neural Network_, dirancang khusus untuk menganalisis dan memahami pola-pola kompleks dalam data deret waktu seperti harga saham.

Dengan memanfaatkan data historis, prediksi yang akurat mengenai pergerakan harga saham BCA tidak hanya memberikan pandangan yang lebih baik bagi investor untuk mengambil keputusan yang tepat, tetapi juga membantu perusahaan dalam merumuskan strategi bisnis yang lebih efektif. Dengan demikian, pendekatan ini dapat berperan penting dalam mengurangi risiko kerugian finansial dan meningkatkan kinerja investasi secara keseluruhan.

## Business Understanding

### Problem Statements

- Prediksi harga saham merupakan tugas yang sangat kompleks dan selalu melibatkan risiko. Bagaimana risiko ini dapat diminimalkan?
- Bagaimana data historis dapat digunakan untuk memprediksi pergerakan harga saham di masa mendatang?

### Goals

- Mengembangkan model _Machine Learning_ yang dapat memprediksi harga saham dengan kesalahan yang rendah.
- Memprediksi harga saham untuk 30 hari kedepan dari tanggal 21 Februari 2024 s/d 21 Maret 2024.

### Solution statements

- Model _Long Short-Term Memory (_LSTM_)_ dibuat dengan menggunakan library TensorFlow dan PyTorch untuk melakukan prediksi harga saham. _LSTM_ merupakan jenis _Recurrent Neural Network_ yang dirancang untuk mempelajari dependensi jangka panjang, yang sangat berguna untuk prediksi data deret waktu seperti harga saham.
- Pengoptimalan hiperparameter pada model _LSTM_ dilakukan dengan menggunakan library KerasTuner dan Optuna untuk meningkatkan akurasi prediksi. Metrik evaluasi yang digunakan adalah _Mean Squared Error (MSE)_ dan _Root Mean Squared Error (RMSE)_.

## Data Understanding

Dataset ini berisi data harga saham historis PT Bank Central Asia Tbk (BBCA. JK) dari tahun 8 Juni 2004 hingga 20 Februari 2024. Dataset ini mencakup harga saham harian, volume perdagangan, dan metrik keuangan relevan lainnya untuk bank-bank terkemuka. Harga saham disediakan dalam mata uang IDR (Rupiah Indonesia).

Informasi lebih lanjut terdapat pada [Yahoo finance Saham BCA](https://finance.yahoo.com/quote/BBCA.JK/history/)

### Fitur-fitur yang terdapat pada dataset BBCA.JK:

- _Date_ : Tanggal data harga saham.
- _Open_ : Harga pembukaan saham bank pada tanggal tertentu.
- _Close_ : Harga penutupan saham bank pada tanggal tertentu.
- _High_ : Harga tertinggi yang dicapai oleh saham bank selama hari perdagangan.
- _Low_ : Harga terendah yang dicapai oleh saham bank selama hari perdagangan.
- _Adj Close_ : Harga penutupan pada hari perdagangan tertentu, disesuaikan untuk mencerminkan tindakan korporasi, seperti pemecahan saham, dividen, penawaran hak, atau penyesuaian lain yang dapat mempengaruhi harga saham.
- _Volume_ : Jumlah saham yang diperdagangkan pada tanggal tertentu.
  
  Sumber Data:
  Himpunan data disusun dari sumber keuangan yang andal, termasuk bursa saham, situs web berita keuangan, dan penyedia data keuangan terkemuka. Teknik pembersihan dan _preprocessing_ data telah diterapkan untuk memastikan akurasi dan konsistensi.

### Eksplorasi Data Analysis (EDA)

- Ringkasan statistik deskriptif dari dataset
  
  - count : Jumlah data
  
  - mean : nilai rata-rata
  
  - min : nilai data minimum
  
  - 25% : kuartil pertama (Q1)
  
  - 50% : kuartil kedua (Median)
  
  - 75% : kuartil ketiga (Q3)
  
  - max : nilai data maksimum
  
  - std : standar deviasi

    |       |     Open    |     High     |     Low     |    Close    |  Adj Close  |    Volume    |
    |:-----:|:-----------:|:------------:|:-----------:|:-----------:|:-----------:|:------------:|
    | count | 4881.000000 |  4881.000000 | 4881.000000 | 4881.000000 | 4881.000000 |  4881.000000 |
    |  mean | 3143.235505 |  3172.630608 | 3112.729973 | 3143.271871 | 2886.695823 | 1.078381e+08 |
    |  std  | 2706.217590 |  2727.272606 | 2685.272032 | 2706.073835 | 2662.345316 | 1.302515e+08 |
    |  min  |  175.000000 |  177.500000  |  175.000000 |  177.500000 |  105.656075 | 0.000000e+00 |
    |  25%  |  720.000000 |  725.000000  |  705.000000 |  720.000000 |  553.555847 | 4.898400e+07 |
    |  50%  | 2200.000000 |  2220.000000 | 2180.000000 | 2200.000000 | 1902.267700 | 7.295500e+07 |
    |  75%  | 5315.000000 |  5395.000000 | 5240.000000 | 5295.000000 | 4900.501465 | 1.170345e+08 |
    |  max  | 9975.000000 | 10000.000000 | 9875.000000 | 9950.000000 | 9950.000000 | 1.949960e+09 |

    Tabel 1. Ringkasan statistik deskriptif

- Pengamatan hubungan antar fitur numerik dengan Fungsi _Pairplot_

  Bentuk distribusi ini dapat memberikan informassi tentang hubungan setiap variabel. Pola dalam plot ini dapat menunjukkan korelasi antara variabel. Pada Gambar 1, terlihat banyak fitur yang memiliki nilai sama.
  ![cor plot](https://github.com/maybeitsai/BCA-Stock-Forecasting/assets/130530985/e8cb1b43-b028-43c1-8c2f-2e700894d746)
  Gambar 1. Diagram _Pairplot_

- Pengamatan berdasarkan _Heatmap_ korelasi antar fitur

  Berdasarkan Gambar 2, banyak fitur yang memiliki korelasi tinggi, sedangkan volume memiliki korelasi negatif.

  ![corr](https://github.com/maybeitsai/BCA-Stock-Forecasting/assets/130530985/3e723bb3-3d78-4555-b18b-750a2eb3ef41)
  
  Gambar 2. Diagram _Heatmap_

- Pengamatan _Volume_ dan _Moving Average_

  Perhitungan _Moving Average (MA)_ dari volume perdagangan saham dengan menggunakan jendela (window) 20 hari terakhir. _MA_ volume perdagangan membantu dalam memahami tren atau pola pergerakan volume perdagangan saham selama periode waktu tertentu. Berdasarkan Gambar 3, dapat dilihat bahwa pola pada _Volume_ tidak seimbang.

  ![Volume](https://github.com/maybeitsai/BCA-Stock-Forecasting/assets/130530985/ed18fc9f-f38f-4665-9092-87df187868f5)
  
  Gambar 3. Distribusi _Volume_ dan _Moving Average_

- Analisa fitur yang memiliki korelasi tinggi

  Berdasarkan hasil analisa dari Gambar 4, dapat disimpulkan bahwa fitur-fitur tersebut memiliki nilai yang tidak jauh berbeda.

  ![Price](https://github.com/maybeitsai/BCA-Stock-Forecasting/assets/130530985/2520873c-8310-49ac-a009-081af9c1f14b)
  Gambar 4. Distribusi fitur

- Analisa _Moving Average_ dan _Close Price_

  Penggunaan _Moving Average_ untuk memperhalus data harga penutupan saham selama 50 dan 200 hari terakhir. Dengan memvisualisasikan kedua _Moving Average_ ini bersama harga penutupan sebenarnya, tren jangka pendek dan jangka panjang dalam harga saham BCA dapat terlihat. Berdasarkan analisa dari Gambar 5, dapat dilihat grafik dari harga penutupan selalu naik walaupun tidak stabil.

  ![Close](https://github.com/maybeitsai/BCA-Stock-Forecasting/assets/130530985/e7784995-8ad6-495d-8d38-f2922f20ab2a)
  Gambar 5. Distribusi _Moving Average_ dan _Close Price_

## Data Preparation

### Seleksi fitur

Mengambil dua kolom, yaitu kolom "Date" dan "Close", dari DataFrame yang disebut df_analysis. Kolom "Close" ini akan digunakan sebagai kolom target dan diubah menjadi bentuk array satu dimensi menggunakan metode .values.reshape(-1, 1). Hal ini dilakukan untuk memastikan data siap digunakan dalam proses selanjutnya.

### Pembagian dataset 

Dataset dibagi menjadi dua bagian, yaitu data latih dan data uji, dengan menggunakan fungsi _train_test_split_ dari library Sklearn. Data latih digunakan untuk melatih model, sedangkan data uji digunakan untuk menguji kinerja model pada data yang belum pernah dilihat sebelumnya. Dalam proyek ini, data dipisahkan menggunakan metode _train_test_split_ dengan ukuran data uji sebesar 20% dari total data. Penyebutan _shuffle=False_ menunjukkan bahwa data tidak diacak sebelum dipisahkan, yang berarti urutannya dipertahankan. Hal ini penting terutama jika data yang dimiliki memiliki sifat kronologis atau urutan tertentu yang harus dipertahankan dalam pembagian data latih dan uji. Jumlah data latih adalah 3904, sementara jumlah data uji adalah 977.

### Normalisasi

Normalisasi data adalah proses untuk mengubah rentang nilai dari setiap fitur dalam dataset menjadi rentang yang seragam. Dalam proses ini menggunakan _MinMaxScaler_ dari library Sklearn untuk melakukan normalisasi. _MinMaxScaler_ akan mengubah nilai-nilai data sehingga berada dalam rentang antara 0 dan 1.

### Mengubah dimensi dan menentukan variabel x dan y

Pada tahap ini, sebuah fungsi bernama _prepare_data_ dibuat untuk memisahkan variabel x. Setiap 100 data digunakan untuk memprediksi data berikutnya yang menjadi variabel y. Kemudian, dalam fungsi _prepare_data_, fitur (X) dan target (y) diubah menjadi array numpy untuk digunakan dalam model nantinya. Data fitur diubah menjadi format tiga dimensi yang diperlukan oleh model _LSTM_.

## Modeling

Pada tahap ini, Model yang digunakan adalah _Long Short-Term Memory (LSTM)_. Penggunaan model _LSTM_ dalam tahap pemodelan dipilih karena keunggulan-keunggulan tertentu yang dimilikinya, terutama dalam penanganan data deret waktu seperti data keuangan, cuaca, atau bahkan teks. Dua model _LSTM_ dibuat menggunakan dua library yang berbeda, yaitu TensorFlow dan PyTorch. Selain itu, digunakan library KerasTuner dan Optuna untuk menyetel parameter-model.

### LSTM dengan library TensorFlow dan KerasTuner

Model ini merupakan jenis model sekuensial yang umum digunakan dalam jaringan saraf tiruan. Terdiri dari dua lapisan _LSTM_ yang diikuti oleh beberapa lapisan _Dense_. Setiap lapisan memiliki berbagai parameter yang dapat disesuaikan untuk penyesuaian model, seperti jumlah unit, fungsi aktivasi, dll.

Pada lapisan _LSTM_ pertama dan kedua, dilakukan penyetelan hiperparameter dengan jumlah unit antara 16 hingga 256, dengan langkah 16. Lapisan _LSTM_ pertama berfungsi sebagai lapisan _input_, dengan _return_sequences=True_ untuk mengembalikan urutan output lengkap dari setiap langkah waktu. Hal ini berguna ketika lapisan _LSTM_ diikuti oleh lapisan _LSTM_ lain atau lapisan lain yang memerlukan urutan _output_ dari setiap langkah waktu.

Setelah dua lapisan _LSTM_, model menggunakan beberapa lapisan _Dense_ untuk memproses fitur yang dihasilkan oleh lapisan _LSTM_ sebelumnya. Dalam model ini, terdapat tiga lapisan _Dense_. Lapisan _Dense_ pertama menggunakan parameter _units_ yang disesuaikan dengan fungsi hiperparameter, dengan jumlah unit antara 16 hingga 128 dan langkah 16. Lapisan _Dense_ kedua menggunakan parameter units yang disesuaikan dengan fungsi hiperparameter, dengan jumlah unit antara 8 hingga 64 dan langkah 8. Lapisan _Dense_ terakhir memiliki satu neuron, bertanggung jawab untuk menghasilkan output akhir dari model, yaitu prediksi yang diinginkan.

Setelah parameter yang ingin diatur ditetapkan, selanjutnya melakukakan _compile_ dengan menggunakan _Optimizer Adam_ dengan _learning rate_ yang disetel juga dengan nilai 0.01, 0.001, dan 0.0001.

Pada model ini juga menggunakan beberapa fungsi _Callback_ yaitu :
- _EarlyStopping_ : berfungsi untuk menghentikan pelatihan lebih awal jika metrik yang dipantau tidak meningkat setelah sejumlah _epoch_ tertentu _(patience)_ dan mengembalikan bobot model ke iterasi terbaik selama pelatihan.
  
- _ModelCheckPoint_ : berfungsi untuk menyimpan model ke dalam file best_model.h5 hanya jika nilai metrik yang dipantau terbaik dari semua _epoch_ yang telah dilalui dan memastikan bahwa hanya model dengan performa terbaik yang disimpan.

- _ReduceLROnPlateau_ : Mengurangi laju pembelajaran _(learning rate)_ jika tidak ada peningkatan dalam metrik yang dipantau setelah sejumlah _epoch_ tertentu _(patience)_.

- _rmse_threshold_callback_ : bertujuan untuk menghentikan proses pelatihan model jika nilai _RMSE (Root Mean Squared Error)_ pada data latih dan validasi sudah mencapai batas tertentu.

#### Pengoptimalan hiperparameter dengan _RandomSearch_
Metode pengoptimalan hiperparameter yang digunakan adalah _RandomSearch_, yang menggunakan pendekatan acak untuk mencari kombinasi hiperparameter yang optimal. Dalam _RandomSearch_, nilai untuk setiap hiperparameter dipilih secara acak dari rentang yang telah ditentukan, kemudian kinerja model dievaluasi untuk setiap kombinasi tersebut.

Langkah-langkah pengoptimalan hiperparameter sebagai berikut:

- Pembuatan _Tuner_ : Objek _Tuner_ dibuat menggunakan metode _RandomSearch_ dari KerasTuner akan secara acak mencari kombinasi hiperparameter. Parameter yang diberikan untuk objek _Tuner_ mencakup:
  - _build_model_ : Fungsi yang digunakan untuk membangun model.
  - _objective_ : Menunjukkan apakah tujuan adalah meminimalkan atau memaksimalkan metrik yang ditentukan. Dalam kasus ini, tujuannya adalah meminimalkan nilai kesalahan.
  - _max_trials_ : Jumlah maksimum percobaan pencarian hiperparameter yang akan dilakukan. Dalam contoh ini, hingga 50 kombinasi hiperparameter akan dicoba.
  - _executions_per_trial_ : Jumlah kali model dievaluasi untuk setiap konfigurasi hiperparameter. Dalam contoh ini, setiap model dievaluasi hanya sekali.
  - _directory_ dan _project_name_ : Direktori tempat _Tuner_ akan menyimpan hasil pencarian.
  
- Pencarian : Setelah _Tuner_ dibuat, metode pencarian digunakan untuk memulai proses pencarian hiperparameter. Parameter yang diberikan meliputi:
  - _X_train_ dan _y_train_ : Data latih dan label.
  - _epochs_ : Jumlah _epochs_ yang akan digunakan untuk melatih setiap model yang diuji.
  - _batch_size_ : Ukuran _batch_ yang digunakan selama pelatihan.
  - _validation_data_ : Data validasi yang digunakan untuk mengevaluasi kinerja model.

### LSTM dengan library PyTorch dan Optuna

Terdapat beberapa proses dalam pembuatan model :

-  Pembangunan Model _LSTM_ : Kelas _LSTM_ Model didefinisikan sebagai implementasi jaringan _LSTM_ dalam PyTorch. Model ini memiliki beberapa parameter, termasuk ukuran input, ukuran tersembunyi, jumlah lapisan, dan ukuran output.

- Pembuatan Fungsi Evaluasi Model : Fungsi ini berguna untuk mengevaluasi kinerja model selama fase validasi.

- Pembuatan _Callback_ : Bertujuan untuk mencegah _overfitting_, mekanisme _Early Stopping_ digunakan dengan definisi kelas _EarlyStopping_.

- Pelatihan Model : Proses pelatihan dilakukan dengan menggunakan fungsi _train_model_, di mana model dievaluasi pada setiap _epoch_ untuk mengamati kinerjanya pada data pelatihan dan validasi.

#### Pengoptimalan Hiperparameter dengan menggunakan Optuna
Optuna digunakan untuk mengoptimalkan hiperparameter model, seperti ukuran tersembunyi _(hidden size)_, jumlah lapisan _(num layers)_, dan tingkat pembelajaran _(learning rate)_. Optuna adalah library yang memanfaatkan algoritma pencarian berbasis pohon dan teknik _Pruning_ untuk mencari hiperparameter yang optimal.

### Model Selection

#### Model Pertama : Menggunakan TensorFlow dan KerasTuner

##### Kelebihan :

- Kemudahan Penggunaan : TensorFlow dan KerasTuner menyediakan antarmuka yang mudah digunakan dan dokumentasi yang lengkap, membuatnya cocok untuk pengguna dari berbagai tingkat keahlian.
- KerasTuner : KerasTuner menyediakan alat yang kuat untuk penyetelan hiperparameter secara otomatis, yang dapat menghemat waktu dan upaya dalam proses penyetelan model.
- Integrasi yang Kuat: TensorFlow memiliki integrasi yang kuat dengan banyak alat dan platform lain dalam ekosistem _Machine Learning_, sehingga memudahkan untuk melakukan visualisasi, pengembangan, dan penggunaan model.
- Performa : TensorFlow telah dikenal memiliki kinerja yang baik, terutama dalam konteks pelatihan model pada data dalam skala besar.

##### Kekurangan :

- Fleksibilitas Terbatas : Keras, sebagai _high-level API_ dalam TensorFlow, memiliki keterbatasan dalam fleksibilitas dan kustomisasi jika operasi yang sangat khusus diperlukan atau tidak didukung secara langsung oleh Keras.
- Kurangnya Kontrol Detail : Meskipun TensorFlow memberikan tingkat abstraksi yang tinggi, ini juga berarti beberapa kontrol detail mungkin hilang dibandingkan dengan pendekatan yang lebih rendah seperti PyTorch.

#### Model Kedua : Menggunakan PyTorch dan Optuna

##### Kelebihan :

- Fleksibilitas dan Kontrol : PyTorch memungkinkan tingkat fleksibilitas dan kontrol yang lebih tinggi dalam merancang dan menyesuaikan arsitektur model. Ini membuatnya cocok untuk penelitian yang lebih eksploratif dan pengembangan model yang canggih.
- _Dynamic Computational Graph_ : PyTorch menggunakan graph komputasi dinamis yang memudahkan dalam menyesuaikan arsitektur model dan melakukan _debugging_.
- Optuna : Optuna menyediakan alat yang kuat untuk penyetelan hiperparameter dengan berbagai algoritma pencarian, memberikan fleksibilitas dalam menyesuaikan strategi penyetelan sesuai dengan kebutuhan proyek.

##### Kekurangan :

- Kurva Pembelajaran : PyTorch mungkin memiliki kurva pembelajaran yang lebih tinggi bagi pengguna yang tidak terbiasa dengan paradigma graph komputasi dinamis atau yang datang dari latar belakang penggunaan TensorFlow.
- Kekurangan Dokumentasi : Meskipun PyTorch telah meningkatkan dokumentasinya dalam beberapa tahun terakhir, beberapa pengguna masih menganggap dokumentasi TensorFlow lebih lengkap dan mudah diakses.
- Kurangnya Integrasi : Meskipun PyTorch mulai memiliki integrasi dengan beberapa alat dan platform lain, integrasinya belum sekuat TensorFlow dalam beberapa aspek seperti deployment dan penggunaan skala besar.

## Evaluation

### Metrik

- _Mean Squared Error (MSE)_ :
_Mean Squared Error_ adalah metrik yang digunakan untuk mengukur seberapa dekat rata-rata kuadrat dari selisih antara nilai yang diprediksi dan nilai yang sebenarnya dari data sampel. Metrik ini digunakan pada model pertama dan kedua. Formula untuk _MSE_ adalah sebagai berikut :

  _MSE_ = $\frac{1}{n} \Sigma_{i=1}^n({y}-\hat{y})^2$

  Keterangan :

    n : jumlah sampel

    Y : nilai sebenarnya dari sampel ke-i

    Ŷ : nilai yang diprediksi untuk sampel ke-i

- _Root Mean Squared Error (RMSE)_ :
_Root Mean Squared Error_ adalah akar kuadrat dari MSE. Ini memberikan ukuran kesalahan rata-rata antara nilai yang diprediksi dan nilai yang sebenarnya dalam satuan yang sama dengan variabel target. Metrik ini digunakan pada model pertama saja. _RMSE_ dihitung dengan cara berikut :

  _RMSE_ = $\sqrt{MSE}$

### Model Evaluation
- Model Pertama
Pada model pertama menghasilkan nilai kesalahan sebagai berikut :
    - _MSE_ : 2.6599e-05 atau 0.000026599
    - _Validation MSE_ : 3.4602e-04 atau 0.00034602
    - _RMSE_ : 0.0052
    - _Validation RMSE_ : 0.0186
 
- Model Kedua
Pada model kedua menghasilkan nilai kesalahan sebagai berikut :
    - _MSE_ : 0.000041
    - _Validation MSE_ : 0.002514

### Simulasi
Dari hasil perbandingan _MSE_ antara kedua model, terbukti bahwa model pertama lebih efektif dalam menangani kesalahan. Oleh karena itu, akan dilakukan simulasi prediksi harga saham BCA untuk 30 hari ke depan menggunakan Model Pertama.

Berikut adalah hasil prediksi harga saham BCA 30 hari kedepan mulai dari tanggal 21 Februari 2024 s/d 21 maret 2024.

![image](https://github.com/maybeitsai/BCA-Stock-Forecasting/assets/130530985/823ca839-4dce-46ea-a523-22301e42f634)

Gambar 6. Prediksi harga saham BCA 30 hari kedepan

### Kesimpulan
#### Goals Achievement
Tujuan proyek yang telah ditetapkan sebelumnya adalah untuk mengembangkan model _Machine Learning_ terbaik yang dapat memprediksi harga saham dengan kesalahan yang rendah untuk PT Bank Central Asia Tbk (BCA). Dengan menggunakan model _LSTM_ dengan pengoptimalan hiperparameter menggunakan TensorFlow dan KerasTuner, serta evaluasi menggunakan metrik _Mean Squared Error (MSE)_ dan _Root Mean Squared Error (RMSE)_, dapat disimpulkan bahwa proyek ini berhasil mencapai tujuan tersebut.

#### Solusi Efektif
Penggunaan model _LSTM_ yang diimplementasikan menggunakan TensorFlow dan KerasTuner, serta PyTorch dan Optuna, terbukti efektif dalam memprediksi harga saham BCA. Dengan hasil evaluasi yang menunjukkan kesalahan yang rendah, dapat disimpulkan bahwa solusi yang diusulkan mampu menjadi solusi yang efektif untuk menyelesaikan permasalahan memprediksi pergerakan harga saham.

#### Perluasan Penelitian
Meskipun model pertama menunjukkan performa yang lebih baik dalam menangani kesalahan, ada potensi untuk melakukan perbaikan lebih lanjut atau penelitian lanjutan untuk meningkatkan kinerja model. Misalnya, eksplorasi lebih lanjut dalam pengaturan hiperparameter atau penerapan teknik lain dari _Deep Learning_ seperti penggunaan Transformer.

#### Hasil Simulasi
Berdasarkan analisa dari Gambar 6, dapat diperkirakan harga saham BCA 30 hari kedepan akan turun disekitar harga 9800. Simulasi prediksi harga saham BCA 30 hari ke depan menggunakan Model Pertama dapat memberikan pandangan yang lebih baik bagi investor untuk mengambil keputusan yang tepat. Namun, tetap perlu diingat bahwa prediksi harga saham memiliki tingkat ketidakpastian, dan hasilnya perlu diperhatikan bersama dengan faktor-faktor lain dalam pengambilan keputusan investasi.

Dengan demikian, dapat disimpulkan bahwa proyek _Machine Learning_ ini berhasil mencapai tujuan yang telah ditetapkan dan memberikan solusi yang efektif dalam memprediksi harga saham BCA.

## Referensi

[1] Brownlee, J. (2019). Deep Learning for Time Series Forecasting. Machine Learning Mastery.

[2] TensorFlow Documentation. (2022). TensorFlow. Diakses dari https://www.tensorflow.org/api_docs

[3] PyTorch Documentation. (2022). PyTorch. Diakses dari https://pytorch.org/docs/stable/index.html

[4] Optuna Documentation. (2022). Optuna. Diakses dari https://optuna.org/

[5] KerasTuner Documentation. (2022). KerasTuner. Diakses dari https://keras.io/api/keras_tuner/

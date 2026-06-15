# Laporan Proyek Machine Learning - Zalfa Alfiah Anasya (2330511015) & Nazwa Syafiya Mirzal (2330511032)

## Project Overview

![Ilustrasi Polusi Udara](https://images.unsplash.com/photo-1524661135-423995f22d0b?q=80&w=1000&auto=format&fit=crop)

Pencemaran udara merupakan permasalahan lingkungan yang kritis di DKI Jakarta akibat tingginya mobilitas dan aktivitas industri, sehingga masyarakat dan pemangku kebijakan sangat membutuhkan sistem peringatan dini yang otomatis dan mudah dipahami. Proyek ini bertujuan untuk membangun model klasifikasi cerdas yang mampu memprediksi kategori Indeks Standar Pencemaran Udara (ISPU) secara otomatis berdasarkan input parameter gas polutan tanpa memerlukan perhitungan manual. Mengacu pada standar metodologi CRISP-DM (Cross-Industry Standard Process for Data Mining), pengembangan sistem ini menggunakan algoritma machine learning *Random Forest* untuk mencapai tingkat akurasi yang tinggi dalam mengenali pola klasifikasi yang kompleks, guna mendukung tindakan mitigasi yang cepat dan pengambilan keputusan yang tepat di bidang kesehatan masyarakat.

💡 Manfaat Proyek:

✔ Membantu masyarakat dan pemangku kebijakan memantau kelayakan udara secara cepat dan otomatis.

✔ Mempermudah proses evaluasi lingkungan tanpa harus menghitung indeks polutan secara manual.

✔ Menggunakan algoritma Random Forest untuk menghasilkan prediksi klasifikasi Indeks Standar Pencemaran Udara (ISPU) yang lebih akurat dan dapat diandalkan.

  Format Referensi: [KLASIFIKASI KUALITAS UDARA DKI JAKARTA](https://jurnal.globalscients.com/index.php/jiki/article/view/993)

## Business Understanding

**Pernyataan Masalah (Problem Statements)**
* Bagaimana merancang sebuah model klasifikasi prediktif yang mampu mengidentifikasi status kualitas udara DKI Jakarta secara otomatis berbekal data parameter gas polutan harian?
* Seberapa tinggi tingkat akurasi dan efektivitas algoritma *Random Forest* saat diterapkan untuk memprediksi kategori Indeks Standar Pencemaran Udara (ISPU)?

**Tujuan Proyek (Project Goals)**
* Membangun dan melatih model *machine learning* yang bisa memprediksi kategori kelayakan udara secara mandiri, sehingga proses pemantauan lingkungan tidak lagi bergantung pada perhitungan manual.
* Menguji dan mengevaluasi kinerja algoritma *Random Forest* (serta membandingkannya dengan algoritma eksplorasi lain) untuk memastikan model yang dihasilkan cukup akurat sebagai acuan peringatan dini bagi masyarakat.

**Pendekatan Solusi (Solution Approach)**
* **Pendekatan Metodologi:** Menjalankan siklus pengolahan data menggunakan kerangka kerja standar industri CRISP-DM, mulai dari pemahaman awal (*Business & Data Understanding*), pembersihan data (*Data Preparation*), hingga siap dievaluasi dan digunakan.
* **Pendekatan Algoritma:** Mengimplementasikan **Random Forest Classifier** sebagai model utama karena keandalannya dalam menangani data tabular dan klasifikasi multikelas. Sebagai tambahan nilai eksplorasi, model dasar ini juga akan diukur dan dibandingkan dengan algoritma *ensemble* tingkat lanjut seperti XGBoost guna mencari performa klasifikasi paling optimal.

## Data Understanding

Dataset utama yang dieksplorasi dalam proyek ini bersumber dari repositori data publik Kaggle, yang dapat diakses melalui tautan berikut: [Dataset ISPU DKI Jakarta](https://www.kaggle.com/datasets/senadu34/air-quality-index-in-jakarta-2010-2021). Dataset terdiri dari 5.538 baris data pemantauan harian dari 5 stasiun di DKI Jakarta pada rentang waktu 1 Januari 2010 hingga 28 Februari 2025. 

Tahap ini berfokus pada pengenalan dan pemahaman terhadap struktur data yang digunakan dalam penelitian. Berbeda dengan sistem yang membutuhkan penggabungan banyak tabel, seluruh informasi mengenai Indeks Standar Pencemaran Udara (ISPU) di wilayah DKI Jakarta ini sudah terintegrasi secara praktis ke dalam satu berkas data tabular tunggal bernama `ispu_dki_all.csv`.

Proses pemuatan data ke dalam lingkungan kerja Google Colab dilakukan dengan menggunakan baris kode berikut:

```python
df = pd.read_excel('/content/drive/MyDrive/ML/ispu_dki_all.csv.xlsx')
```

📂 **Informasi Dataset:**
| kolom    | Tipe Data  | Keterangan  |
|----------|-------------|-----------------|
| tanggal  | datetime     | Tanggal pengukuran |
| stasiun  | string     | Nama stasiun pemantauan |
| pm25     | float     | Konsentrasi PM2.5 (µg/m³) |
| pm10     | float     | Konsentrasi PM10 (µg/m³) |
| so2      | float     | Konsentrasi SO2 (µg/m³) |
| co       | float     | Konsentrasi CO (µg/m³) |
| 03       | float     | Konsentrasi O3 (µg/m³) |
| no2      | float     | Konsentrasi NO2 (µg/m³) |
| max      | 340.556     | Nilai polutan tertinggi pada hari tersebut |
| critical | 271.360     | Polutan yang menjadi penentu kategori |
| categori | 271.360     | Kategori kualitas udara (label) |

📌 **Uraian Fitur:**
- **pm25**: Partikel halus berdiameter < 2.5 mikrometer, berbahaya bagi saluran pernapasan.
- **pm10**: Partikel berdiameter < 10 mikrometer.
- **so2**: Sulfur dioksida, berasal dari pembakaran bahan bakar fosil.
- **co**: Karbon monoksida, gas tidak berwarna hasil pembakaran tidak sempurna.
- **o3**: Ozon troposfer, terbentuk dari reaksi kimia polutan lain.
- **no2**: Nitrogen dioksida, berasal dari emisi kendaraan bermotor.
- **categori**: Label target — BAIK, SEDANG, TIDAK SEHAT, SANGAT TIDAK SEHAT, BERBAHAYA.

🔍 **Kondisi Data:**

1. Missing Values
```python
missing = pd.DataFrame({
    'Jumlah': df.isnull().sum(),
    'Persentase (%)': (df.isnull().sum() / len(df) * 100).round(2)
})
print(missing)
```
| kolom    | Jumlah Missing | Persentase (%)  |
|----------|-------------|-----------------|
| stasiun  | 1     | 0.02 |
| pm25     | 4022     | 72.63 |
| pm10     | 315     | 5.69 |
| so2      | 130     | 2.35 |
| co       | 88     | 1.59 |
| 03       | 104     | 1.88 |
| no2      | 106     | 1.91 |
| max      | 1     | 0.02 |
| critical | 4     | 0.07 |

Kolom pm25 memiliki missing value sangat tinggi (72%) sehingga tidak digunakan sebagai fitur dalam pemodelan.

2. Duplikat Data

```python
print('Jumlah duplikat:', df.duplicated().sum())
```
Tidak ditemukan baris duplikat pada dataset.

3. Distribusi Label
```python
print(df['categori'].value_counts())
```
| Kategori    | Jumlah| 
|----------|-------------|
| SEDANG  | 3187     | 
| TIDAK SEHAT     | 1827     | 
| BAIK     | 316     | 
| SANGAT TIDAK SEHAT      | 203     | 
| TIDAK ADA DATA       | 4     | 
| BERBAHAYA       | 1     | 

🔍 Exploratory Data Analysis (EDA)

✔ Distribusi kategori menunjukkan dominasi kelas SEDANG dan TIDAK SEHAT, mengindikasikan adanya ketidakseimbangan kelas yang perlu ditangani.

![Distribusi kategori kualitas udara](https://github.com/user-attachments/assets/f794c137-4a09-4357-b777-06d997b768b1)

✔ Distribusi nilai tiap polutan divisualisasikan menggunakan histogram untuk melihat sebaran dan kemungkinan outlier.

![Distribusi nilai polutan](https://github.com/user-attachments/assets/0aceafb2-1629-45d3-9b15-64311d72ef22)

✔ Heatmap korelasi antar polutan digunakan untuk memahami hubungan antar fitur sebelum pemodelan.

![Heatmap nilai polutan](https://github.com/user-attachments/assets/e424e6ed-9a25-42f0-9105-f6565cb29f70)

## Data Preparation
Proses data preparation dilaksanakan melalui beberapa tahapan untuk menjamin kualitas dan kesiapan dataset sebelum memasuki tahap pemodelan.

📌 **Step 1 — Hapus Label Tidak Relevan**
Baris dengan label TIDAK ADA DATA (4 baris) dihapus karena bukan merupakan kategori kualitas udara yang valid, melainkan indikasi bahwa sensor tidak merekam data pada hari tersebut.
```python
df = df[df['categori'] != 'TIDAK ADA DATA']
```
📌 **Step 2 — Penggabungan Kelas Minor**
Kelas SANGAT TIDAK SEHAT (203 data) dan BERBAHAYA (1 data) digabungkan ke dalam kelas TIDAK SEHAT. Penggabungan dilakukan karena:
- Kelas BERBAHAYA hanya memiliki 1 sampel — tidak cukup untuk dipelajari model
- Secara makna, ketiganya sama-sama menggambarkan kondisi udara yang merugikan kesehatan
- Penggabungan menghasilkan 3 kelas yang lebih representatif untuk proses training
```python
mapping_label = {
    'BAIK': 'BAIK',
    'SEDANG': 'SEDANG',
    'TIDAK SEHAT': 'TIDAK SEHAT',
    'SANGAT TIDAK SEHAT': 'TIDAK SEHAT',
    'BERBAHAYA': 'TIDAK SEHAT'
}
df['label'] = df['categori'].map(mapping_label)
```
Distribusi label setelah penggabungan:
| Label    | Jumlah| 
|----------|-------------|
| SEDANG  | 2829     | 
| TIDAK SEHAT     | 1770     | 
| BAIK     | 276     | 

📌 **Step 3 — Seleksi Fitur**
Kolom pm25 tidak digunakan karena 72% datanya kosong. Fitur yang digunakan adalah: pm10, so2, co, o3, no2.
```python
fitur = ['pm10', 'so2', 'co', 'o3', 'no2']
df_model = df[fitur + ['label']].dropna()
```
Setelah menghapus missing values pada fitur yang digunakan, dataset final berjumlah 4.875 baris.

📌 **Step 4 — Split Data Training dan Testing**
Dataset dibagi dengan rasio 80:20 menggunakan stratify=y untuk memastikan proporsi tiap kelas tetap seimbang di data training dan testing.
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```
|     | | 
|----------|-------------|
| Data Training  | 3900 baris     | 
| Data Testing     | 975 baris     |

## Modeling
📝 **Algoritma: Random Forest Classifier**
Random Forest adalah metode ensemble learning yang membangun sejumlah pohon keputusan (decision tree) secara paralel menggunakan teknik bagging (Bootstrap Aggregating). Setiap pohon dilatih dari sampel data yang berbeda-beda, kemudian prediksi akhir ditentukan berdasarkan voting mayoritas dari seluruh pohon. Pendekatan ini membuat Random Forest lebih robust dan akurat dibanding pohon keputusan tunggal.
```python
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)
```
**Parameter yang digunakan:**
| Parameter    | Nilai | Alasan  |
|----------|-------------|-----------------|
| n_estimators  | 100     | Jumlah pohon yang dibangun; semakin banyak umumnya semakin stabil |
| random_state     | 42     | Memastikan hasil yang reproducible |
| class_weight     | balanced     | Menangani ketidakseimbangan kelas agar kelas minoritas (BAIK) tetap dipelajari dengan baik |

📌 **Feature Importance**
Setelah training, Random Forest menghasilkan nilai feature importance yang menunjukkan seberapa besar kontribusi tiap polutan dalam menentukan prediksi kategori kualitas udara.
```python
importances = pd.Series(rf_model.feature_importances_, index=fitur).sort_values(ascending=False)
```
![Feature Importance](https://github.com/user-attachments/assets/995a91e1-396f-4bb8-ac2d-7ea8c3a0d4e8)

## Evaluation
**Metrik Evaluasi yang Digunakan:**
- **Accuracy —** persentase prediksi yang benar dari keseluruhan data testing
- **Precision —** dari semua prediksi suatu kelas, berapa yang benar-benar kelas tersebut
- **Recall —** dari semua data yang sebenarnya suatu kelas, berapa yang berhasil terdeteksi
- **F1-Score —** rata-rata harmonis Precision dan Recall, lebih representatif untuk data tidak seimbang
```python
y_pred = rf_model.predict(X_test)
print(f'Akurasi: {accuracy_score(y_test, y_pred):.4f}')
print(classification_report(y_test, y_pred))
```
**Hasil Evaluasi:**

| Kelas    | Precision | Recall  | F1-Score| Support|
|----------|-------------|-----------------| -----------------| -----------------|
| BAIK  | 0.94     | 0.84 | 0.88| 55|
| SEDANG    | 0.95     |  0.98| 0.96| 566|
| TIDAK SEHAT     | 0.97     |  0.94| 0.95| 354|
| Accuracy  |      |  | 0.95| 975|

**Confusion Matrix** menunjukkan detail prediksi benar dan salah per kelas. Nilai diagonal merepresentasikan prediksi yang benar, sedangkan nilai di luar diagonal merepresentasikan kesalahan klasifikasi.

![Confussion Matrix](https://github.com/user-attachments/assets/07e07f24-0e7a-4f01-afc8-57dd45ef813d)

📌 **Kesimpulan Evaluasi:**

✔ Model Random Forest berhasil mengklasifikasikan kategori kualitas udara dengan akurasi yang baik.

✔ Parameter class_weight='balanced' terbukti membantu model dalam mengenali kelas BAIK yang jumlah datanya lebih sedikit.

✔ Fitur yang paling berpengaruh berdasarkan feature importance adalah [isi setelah jalankan notebook], menunjukkan bahwa polutan tersebut menjadi penentu utama kategori kualitas udara di DKI Jakarta.

# Kesimpulan

✔ Model Random Forest berhasil dibangun untuk mengklasifikasikan kualitas udara DKI Jakarta ke dalam 3 kategori: BAIK, SEDANG, dan TIDAK SEHAT.

✔ Fitur pm25 tidak diikutsertakan dalam pemodelan karena memiliki missing value sebesar 72%, sementara fitur pm10, so2, co, o3, dan no2 digunakan sebagai prediktor.

✔ Penggabungan kelas SANGAT TIDAK SEHAT dan BERBAHAYA ke dalam TIDAK SEHAT dilakukan untuk menangani data yang sangat tidak seimbang pada kelas minoritas.

✔ Model ini dapat dikembangkan lebih lanjut dengan menambahkan data pm25 jika tersedia, atau mengeksplorasi metode penanganan imbalance seperti SMOTE untuk meningkatkan performa pada kelas BAIK.

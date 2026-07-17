# Arsitektur Prediksi Tinggi Muka Air

## 1. Tujuan dan batasan

Pipeline memprediksi `tma_mdpl` untuk 30 pos pada 19 September 2025 sampai 18 Mei 2026, pukul 06.00, 12.00, dan 18.00. Horizon ini terdiri dari 242 hari atau 21.780 prediksi. Kode berada di `notebook-150726.ipynb`. Rolling validation V2 telah dijalankan; full final fit sengaja diserahkan ke Kaggle.

Prinsip desain utamanya adalah kesetaraan antara kondisi validasi dan deployment. Karena tidak ada TMA aktual setelah 18 September 2025, model utama tidak memakai lag TMA sebagai input. Lag autoregresif terlihat kuat pada validasi satu langkah, tetapi untuk horizon delapan bulan harus memakai prediksinya sendiri dan akan mengakumulasi error serta gagal saat rezim hidrologi berubah.

## 2. Ringkasan audit data

| Sumber | Ukuran/coverage | Temuan penting |
|---|---:|---|
| Target train | 84.396 baris, 30 pos | 1 Januari 2023 06.00 - 18 September 2025 18.00; tidak ada `NaN` atau key duplikat |
| Grid target teoritis | 89.280 baris | 4.884 timestamp tidak tercatat sebagai baris, terutama 4-28 Februari 2025; Gunungsari baru tersedia sejak 13 Agustus 2024 |
| Test/submission | 21.780 baris | ID dan urutan `test.csv` identik dengan sample submission |
| Lingkungan | 888.480 baris, 27 kolom | Grid per jam lengkap untuk 30 pos dari 1 Januari 2023 sampai 18 Mei 2026 |
| Koordinat | 30 baris | Set nama pos persis sama dengan train/test/lingkungan |
| HydroRIVERS | 836.472 reach pada tile AU | CRS WGS84; 15 atribut sesuai dokumentasi |

Temuan kualitas khusus:

- `rainfall_mm` dan `rainfall_openmeteo_mm` identik pada seluruh 888.480 baris. Hanya satu yang digunakan agar model tidak memberi bobot ganda pada sinyal yang sama.
- `solar_radiation_mj_m2` memiliki 100.080 nilai sentinel `-999`, seluruhnya pada masa mendatang. Sentinel diubah menjadi missing dan disertai flag; nilai negatif tidak diperlakukan sebagai radiasi nyata.
- Pada 18 Mei 2026, enam variabel soil/pressure dan lima variabel MJO kosong selama 24 jam. Nilainya di-forward-fill secara kausal maksimum 24 jam dan flag missing dipertahankan.
- `nino_34` kosong pada 1-18 Mei 2026. Karena indeks ini berubah lambat/bulanan, digunakan causal forward-fill maksimal 31 hari plus flag missing.
- Ada lonjakan TMA satu-titik yang tidak fisik, misalnya Napel 34,95 -> 325,83 -> 34,55. Titik seperti ini tidak dihapus; quality mask V2 memberi bobot 0,05-0,20 berdasarkan empat nearest temporal neighbors, robust scale pos, dan batas gap 45 hari. Kejadian tinggi yang berlangsung beberapa timestamp tetap berbobot penuh.

## 3. Model yang dipilih

Model utama V2 adalah ensemble tiga keluarga `CatBoostRegressor` global: direct Huber anchor, seasonal-residual Huber, dan seasonal-residual RMSE. Setiap keluarga memakai seed 17, 41, dan 83. Direct anchor menggunakan target:

```text
z(pos, t) = (TMA(pos, t) - median_train(pos)) / max(IQR_train(pos) / 1,349; 0,05)
```

Model residual menggunakan `(TMA - baseline_musiman_crossfit) / scale_pos`. Prediksi seed dirata-ratakan per keluarga, dikembalikan ke meter, lalu diblend dengan bobot rolling-OOF. Arsitektur dan bobot lengkap dijelaskan pada Bagian 13.

CatBoost dipilih karena jumlah label sedang, terdapat interaksi nonlinier kuat, fitur kategorikal/statis, missing value, dan distribusi yang tidak Gaussian. Normalisasi robust mencegah pos berelevasi absolut tinggi mendominasi training. Huber memberi robustness, sedangkan residual RMSE berbobot `scale_pos^2` menyelaraskan optimasi dengan error meter. Direct anchor mencegah model residual terlalu bergantung pada estimasi baseline musiman.

LSTM/Transformer temporal tidak dipilih sebagai arsitektur utama: histori efektif hanya sekitar 2,7 tahun, satu pos memiliki histori jauh lebih pendek, dan rollout 242 hari menimbulkan distribution shift besar. GNN end-to-end juga belum sebanding dengan hanya 30 node dan kurang dari tiga siklus tahunan. Informasi graf tetap digunakan secara eksplisit melalui fitur upstream yang lebih mudah diaudit.

## 4. Aliran data

```text
train/test ID ------> validasi key dan waktu ---------------------------+
                                                                       |
lingkungan per jam --> sanitasi/missing --> fitur rolling --------------+--> direct Huber anchor
                                      |                                |
HydroRIVERS + koordinat --> graf NEXT_DOWN --> upstream ----------------+--> seasonal residual Huber
                                      |                                |               + residual RMSE
kalender + atribut reach statis ----------------------------------------+               |
                                                                                       v
target train --> quality mask --> baseline musiman cross-fit -----------> joint OOF blend
                                                                                       |
                                                                                       v
                                                                                 submission.csv
```

Matriks eksogen tidak memuat lag atau rolling TMA. V2 hanya menambahkan satu fitur level, `seasonal_baseline_v2`, yang dihitung cross-fit untuk training dan hanya dari histori sebelum cutoff untuk validation/test. Guard feature-list tetap menolak fitur target mentah agar tidak terjadi leakage atau recursive rollout.

## 5. Preprocessing

1. ID submission diparse dengan regex timestamp tetap. String tidak di-split secara bebas karena beberapa nama pos mengandung ` - `.
2. Seluruh sumber diperiksa untuk duplikasi `(datetime, nama_pos)`, kesamaan set pos, jam target, urutan sample submission, dan kelengkapan grid lingkungan.
3. Missing baris target tidak diinterpolasi dan tidak dibuat sebagai pseudo-label. Baris berlabel yang benar saja masuk training.
4. Sentinel radiasi `-999` diubah ke `NaN`. CatBoost dapat memakai missing numerik secara native; flag missing tetap menjadi fitur.
5. State lingkungan kosong diisi hanya dengan nilai masa lalu dalam limit yang sesuai. Tidak ada backward-fill, sehingga validasi tidak membaca masa depan.
6. Arah angin diubah menjadi pasangan sinus/kosinus agar 359 derajat dekat dengan 0 derajat.
7. Hujan/radiasi dibatasi pada minimum fisik nol dan soil moisture pada rentang 0-1. Tidak ada clipping berdasarkan statistik test.

## 6. Feature engineering

### 6.1 Dinamis lokal

- Nilai lingkungan saat ini: hujan, kelembapan, suhu, dew point, cloud, angin, soil moisture berlapis, pressure, MJO, Nino 3.4, built surface, dan land cover.
- Dew-point depression dan arah angin sinus/kosinus.
- Akumulasi hujan trailing 6, 24, 72, 168, dan 336 jam.
- Maksimum intensitas 24 jam.
- Antecedent precipitation index berbasis EWM dengan half-life 12, 48, dan 168 jam.
- Rata-rata trailing soil moisture, suhu, kelembapan, dan pressure. Semua window bersifat right-closed/kausal.

### 6.2 Jaringan sungai

Setiap pos dicocokkan dengan reach terdekat di EPSG:32749. Jarak pencocokan disimpan sebagai fitur, bersama confidence `exp(-distance_m / 1000)`. Jarak terbesar yang ditemukan sekitar 1,24 km (Bengkelolor dan Jarum), sehingga confidence penting untuk membatasi ketergantungan buta pada match tersebut.

Dokumentasi HydroRIVERS menjelaskan:

- `NEXT_DOWN`: reach berikutnya di hilir; digunakan untuk traversal graf yang eksak.
- `MAIN_RIV`: reach paling hilir pada jaringan DAS; dipakai sebagai kategori basin.
- `LENGTH_KM`, `DIST_DN_KM`, `DIST_UP_KM`: posisi dan panjang hidraulik.
- `CATCH_SKM`, `UPLAND_SKM`: luas catchment lokal dan total hulu.
- `DIS_AV_CMS`: estimasi debit jangka panjang; bukan pengukuran saat ini.
- `ORD_STRA`, `ORD_CLAS`, `ORD_FLOW`: tiga definisi orde/ukuran sungai.
- `HYBAS_L12`: identitas sub-basin HydroBASINS level 12.

Untuk setiap target pos, kode menelusuri `NEXT_DOWN` dari pos sumber. Jika reach target ditemukan di hilir, kondisi lingkungan sumber menjadi anggota agregat upstream. Bobotnya `exp(-jarak_hidraulik / 75 km)`, kemudian dinormalisasi. Dari sini dibentuk hujan dan soil moisture upstream beserta window/API yang sama dengan fitur lokal.

### 6.3 Kalender dan statis

- Siklus day-of-year dan jam dengan sinus/kosinus.
- Kategori jam dan bulan serta flag musim hujan November-April.
- Latitude/longitude, built surface, land cover.
- Atribut HydroRIVERS mentah dan `log1p` untuk luas catchment, upstream area, debit, serta panjang reach.

## 7. Validasi untuk generalisasi

Validasi menggunakan rolling-origin dengan horizon panjang. Setiap fold hanya melatih data sebelum cutoff dan menguji seluruh rentang berikutnya:

| Fold | Train sampai | Validasi |
|---|---|---|
| sep_2023 | 18 Sep 2023 | 19 Sep 2023 - 18 Mei 2024 |
| may_2024 | 18 Mei 2024 | 19 Mei 2024 - 18 Jan 2025 |
| sep_2024 | 18 Sep 2024 | 19 Sep 2024 - 18 Mei 2025 |
| jan_2025 | 18 Jan 2025 | 19 Jan 2025 - 18 Sep 2025 |

Pos dengan kurang dari 90 label sebelum cutoff tidak dinilai pada fold tersebut. Hal ini mencegah statistik target dari pos yang belum pernah terlihat. Early stopping dilakukan terpisah untuk ketiga komponen; iterasi final setiap komponen adalah median best iteration-nya sendiri.

Metrik yang dicatat:

- MAE dan raw RMSE global;
- quality-weighted RMSE sebagai diagnostic sensor-fault;
- macro-MAE antar-pos agar semua pos mendapat bobot setara;
- worst-station MAE untuk mendeteksi kegagalan lokal;
- persentil ke-95 absolute error;
- pembanding seasonal baseline circular-smoothed per pos-jam-day-of-year.

Model layak dipakai hanya jika perbaikannya konsisten lintas fold, bukan hanya rata-rata pada satu split. Analisis residual sebaiknya dilanjutkan per pos, bulan, intensitas hujan, dan kuantil TMA setelah training benar-benar dijalankan.

## 8. Artefak dan cara menjalankan

Instalasi:

```powershell
python -m pip install -r requirements.txt
```

Setelah membuka `notebook-150726.ipynb`, jalankan semua cell dengan mode default berikut untuk audit tanpa training:

```python
MODE = "audit"
```

Untuk validasi saja, ubah cell konfigurasi mode menjadi:

```python
MODE = "validate"
```

Untuk validasi, final fit, dan submission:

```python
MODE = "all"
```

Final fit dapat dijalankan terpisah memakai `v2_tuning.json` terbaru dari validasi:

```python
MODE = "fit"
```

Pada runtime lokal, output ditulis ke `artifacts/`. Pada Kaggle, output ditulis ke `/kaggle/working/`. Setiap eksekusi training memiliki subdirektori dan ZIP tersendiri berdasarkan waktu mulai run.

## 9. Guardrail operasional

- `MODE = "audit"` adalah default agar Run All tidak melatih model tanpa sengaja.
- CatBoost di-import secara lazy hanya pada mode training.
- Setiap merge memakai validasi cardinality dan submission akhir harus lengkap serta finite.
- Seed, feature list, target scaling, bobot upstream, dan konfigurasi disimpan sebagai artefak.
- Prediksi tidak menggunakan recursive TMA, interpolasi label, backward-fill lingkungan, atau statistik test untuk fit preprocessing.

## 10. Konfigurasi Kaggle

Notebook mendeteksi Kaggle dari keberadaan direktori kompetisi. Path input yang digunakan adalah:

```text
/kaggle/input/competitions/sebelas-maret-statistics-data-science-2026/train.csv
/kaggle/input/competitions/sebelas-maret-statistics-data-science-2026/test.csv
/kaggle/input/competitions/sebelas-maret-statistics-data-science-2026/sample_submission.csv
/kaggle/input/competitions/sebelas-maret-statistics-data-science-2026/data_pendukung/koordinat_pos.csv
/kaggle/input/competitions/sebelas-maret-statistics-data-science-2026/data_pendukung/data_lingkungan.csv
/kaggle/input/competitions/sebelas-maret-statistics-data-science-2026/data_pendukung/HydroRIVERS_v10_au_shp/...
```

Seluruh file hasil ditulis ke `/kaggle/working/`. Jika path kompetisi tidak ditemukan, notebook otomatis kembali memakai `datasets/` dan `artifacts/` untuk eksekusi lokal.

Cell konfigurasi yang disarankan untuk training Kaggle penuh:

```python
MODE = "all"
DATA_DIR = KAGGLE_DATA_DIR
OUTPUT_DIR = Path("/kaggle/working")
ITERATIONS = None
SAVE_FEATURE_MATRICES = True
```

## 11. Handoff training untuk penyempurnaan

Nama run memakai waktu UTC saat pipeline training dimulai:

```text
handoff_YYYYMMDDTHHMMSSZ_<mode>/
handoff_YYYYMMDDTHHMMSSZ_<mode>_in_progress.zip
handoff_YYYYMMDDTHHMMSSZ_<mode>_complete.zip
handoff_YYYYMMDDTHHMMSSZ_<mode>_failed.zip
```

ZIP `in_progress` dibuat setelah feature preparation dan diperbarui setelah setiap fold validasi serta setiap seed final selesai. Setelah sukses, ZIP tersebut diganti oleh `complete`. Jika terjadi exception, artefak parsial, snapshot CatBoost, log, dan traceback dikemas sebagai `failed`.

### 11.1 File inti yang wajib ada

| File | Kegunaan penyempurnaan |
|---|---|
| `manifest.json` | Identitas run, waktu UTC, mode, status, stage, SHA-256 notebook, konfigurasi efektif, bobot blend, iteration count per komponen, sumber tuning, dan hash artefak inti |
| `config.json` | Hyperparameter, seed, definisi fold, iteration request, dan toggle feature matrix |
| `runtime_versions.json` | Versi Python, CatBoost, pandas, NumPy, GeoPandas, dan dependency utama |
| `input_inventory.json` | Path dan ukuran seluruh input termasuk sidecar shapefile |
| `data_audit.json` | Coverage data dan jumlah missing/sentinel sebelum preprocessing |
| `feature_columns.json` | Urutan fitur aktual serta daftar fitur kategorikal |
| `data/target_normalization.csv` | Center dan robust scale setiap pos |
| `data/upstream_station_weights.csv` | Bobot graf sungai yang benar-benar digunakan |
| `data/train_sample_diagnostics.parquet` | Target asli/normalisasi, bobot sampel, dan flag spike |
| `metrics/environment_drift.csv` | Pergeseran distribusi setiap fitur antara train dan test |
| `run.log` | Urutan stage dan pesan pipeline |
| `HANDOFF_README.md` | Indeks singkat isi ZIP |

Jika `SAVE_FEATURE_MATRICES=True`, handoff juga memuat `data/train_features.parquet` dan `data/test_features.parquet`. Kedua file ini sangat disarankan karena memungkinkan Codex menganalisis fitur persis yang masuk model tanpa membangun ulang dataset. Bila Parquet tidak tersedia, pipeline otomatis memakai `csv.gz`.

### 11.2 File dari validasi V1 (run pertama)

- `metrics/oof_predictions.parquet`: datetime, pos, fold, target, prediksi, baseline, target/prediksi ternormalisasi, error, dan absolute error.
- `metrics/validation_metrics.csv`: MAE/RMSE global, macro/worst-station MAE, p95 error, baseline, dan best iteration.
- `metrics/fold_station_metrics.csv`: MAE, RMSE, dan bias setiap pos pada setiap fold.
- `metrics/residual_slices.csv`: residual per pos, bulan, dan jam.
- `metrics/feature_importance.csv`: feature importance setiap fold.
- `models/validation_<fold>.cbm`: model fold untuk inspeksi/reproduksi.
- `data/target_stats_<fold>.csv`: normalisasi target yang hanya di-fit pada training fold.

### 11.3 File dari final fit V1 (run pertama)

- `models/catboost_seed_<seed>.cbm`: tiga anggota ensemble final.
- `metrics/final_feature_importance.csv`: importance setiap seed final.
- `training_progress/final_seed_<seed>/`: log iterasi dan snapshot CatBoost.
- `submission.csv`: output kompetisi dengan urutan ID asli.
- `run_metadata.json` dan `result_summary.json`: ringkasan final iterasi, sumber iterasi, jumlah baris, serta lokasi handoff.

Folder `training_progress/` ditulis langsung oleh CatBoost selama fitting (`learn_error.tsv`, `test_error.tsv`, `catboost_training.json`, `time_left.tsv`, dan snapshot bila tersedia). Penggunaan `train_dir`, `allow_writing_files`, dan snapshot mengikuti [dokumentasi resmi output CatBoost](https://catboost.ai/docs/en/references/training-parameters/output). Untuk penyempurnaan berikutnya, ZIP `complete` atau `failed` inilah yang perlu diberikan kembali kepada Codex. OOF prediction harus menjadi dasar utama keputusan model, bukan training loss.

## 12. Evaluasi run pertama dan arah iterasi kedua

Run `handoff_20260714T231248Z_all` selesai dengan 98 fitur, empat fold, tiga seed final, dan 193 iterasi. Skor leaderboard yang dilaporkan adalah RMSE 1,61079.

### 12.1 Training versus validation

CatBoost run pertama mencatat MAE dan Huber pada target yang dinormalisasi, bukan RMSE dalam meter. Prediksi ketiga model final karena itu dihitung ulang terhadap seluruh train untuk memperoleh perbandingan yang setara:

| Evaluasi | MAE (m) | RMSE (m) | Catatan |
|---|---:|---:|---|
| Train, seluruh label | 0,3575 | 2,2304 | RMSE didominasi spike sensor sangat besar |
| Train, memakai sample weight | 0,3319 | 0,8753 | 37 spike mendapat bobot 0,05 |
| Train, tanpa 37 spike | 0,3306 | 0,7379 | Estimasi in-sample untuk sinyal yang learnable |
| OOF, seluruh fold/label | 0,6226 | 1,6606 | Paling dekat dengan kondisi leaderboard |
| OOF, tanpa spike yang telah ditandai | 0,6109 | 1,1148 | Masih memuat beberapa anomaly yang lolos detektor |
| Leaderboard | - | 1,61079 | Dekat dengan pooled raw OOF: selisih -0,0498 |

Clean train RMSE 0,7379 versus clean OOF RMSE 1,1148 menunjukkan gap generalisasi sekitar 0,377 m atau 51%. Raw train RMSE tidak boleh dipakai untuk menilai overfitting karena beberapa label salah skala/isolated reset mendominasi squared error.

RMSE validation per fold:

| Fold | Best iteration | RMSE model | RMSE baseline |
|---|---:|---:|---:|
| sep_2023 | 42 | 2,6988 | 2,7209 |
| may_2024 | 253 | 0,8605 | 1,1612 |
| sep_2024 | 447 | 1,2267 | 1,6372 |
| jan_2025 | 133 | 1,1906 | 1,3961 |

Fold `sep_2024` paling menyerupai horizon test September-Mei dan merupakan indikator utama, sedangkan `sep_2023` adalah stress test dengan histori training hanya sekitar sembilan bulan. Pooled OOF 1,6606 ternyata sangat dekat dengan leaderboard 1,61079, sehingga validasi secara agregat cukup jujur. Akan tetapi, memilih 193 iterasi bukan penyebab utama gap: pada kurva setiap fold, MAE validation di iterasi 193 hanya 0,001-0,016 lebih buruk dari best iteration masing-masing.

### 12.2 Diagnosis error

- Enam pos pertama menyumbang sebagian besar squared error OOF. Bojonegoro - Kali Kethek sendiri menyumbang 35,9%, terutama karena target isolated 225,13 dan 174,13; Wonogiri Dam menyumbang 14,6% karena perubahan rezim musiman yang berkelanjutan.
- Beberapa anomaly belum ditangkap detektor 12 jam, misalnya Jurug bernilai 0 setelah gap data Februari 2025. Nilai ini menimbulkan absolute error sekitar 81 m pada dua fold yang overlap.
- Setelah spike yang sudah dikenal dikeluarkan, 5% target tertinggi masih memiliki bias -1,508 m dan RMSE 2,008 m. Sebaliknya, separuh target terendah memiliki bias +0,456 m. Model jelas melakukan regresi ke rata-rata: terlalu tinggi pada kondisi rendah dan terlalu rendah pada kondisi tinggi.
- Seed ensemble kurang beragam. Standard deviation prediksi antar-seed pada test rata-rata hanya 0,038 m; menambah lebih banyak seed dengan arsitektur identik kemungkinan memberi gain kecil.
- Fitur kalender menyumbang sekitar 20% importance dan `nino_34` sekitar 9% pada final fit. Test bergeser -0,82 standard deviation pada Nino 3.4 dan sekitar +0,5 standard deviation pada hujan/soil moisture/kelembapan. Ablation diperlukan agar indeks iklim tidak hanya menjadi proxy tahun pada histori yang pendek.
- Wonogiri Dam mempunyai pola reservoir/operasi yang tidak cukup dijelaskan oleh cuaca. Pada awal 2024 model memprediksi sekitar 135 m ketika observasi sekitar 126-127 m. Dibutuhkan baseline musiman spesifik pos dengan recency weighting.

### 12.3 Prioritas eksperimen berikutnya

1. **Selaraskan objective dengan kompetisi tanpa membuang robustness.** Pertahankan model Huber saat ini sebagai anchor. Tambahkan model CatBoost kedua dengan target ternormalisasi tetapi loss RMSE dan bobot `quality_weight * station_scale^2`, sehingga squared error pada skala z ekuivalen dengan RMSE meter. Cari bobot ensemble Huber/RMSE hanya menggunakan OOF.
2. **Perbaiki quality mask target.** Detektor harus membandingkan label dengan rolling median/nearest valid observation sampai +/-7 hari agar anomaly di batas gap seperti Jurug 0 dapat ditandai. Hanya isolated single/short event yang diturunkan bobotnya; banjir yang bertahan beberapa timestamp tidak boleh dihapus. Laporkan raw RMSE dan clean RMSE secara bersamaan.
3. **Dekomposisi baseline-residual.** Bentuk baseline `station x day-of-year x hour` yang circular-smoothed dan recency-weighted, dihitung cross-fit per fold. CatBoost memprediksi residual terhadap baseline, bukan seluruh level absolut. Ini terutama ditujukan untuk Wonogiri Dam dan pos dengan operasi/regime musiman.
4. **Validasi sesuai tujuan.** Gunakan pooled raw RMSE sebagai primary metric, fold `sep_2024` sebagai horizon-aligned metric, clean RMSE sebagai diagnostic, serta high-water p95 RMSE, worst-station RMSE, dan bias sebagai guardrail. Fold awal tetap dipakai sebagai stress test tetapi tidak boleh sendirian menentukan iterasi final.
5. **Tambah diversity, bukan seed identik.** Setelah model RMSE dan baseline-residual tersedia, lakukan OOF stacking/blending. Baru setelah itu pertimbangkan LightGBM residual sebagai challenger. Jangan menambah deep model sebelum baseline ini terlampaui secara konsisten.
6. **Eksperimen hidrologis tahap kedua.** Tambahkan upstream rainfall dengan travel-time shift berdasarkan selisih `DIST_DN_KM` dan beberapa skenario kecepatan aliran. Fitur ini harus lolos ablation pada fold horizon-aligned.
7. **Ablation temporal/climate.** Bandingkan model lengkap dengan versi tanpa Nino/MJO, tanpa fitur statis redundan raw+log, dan dengan calendar-only baseline. Pertahankan fitur hanya bila gain konsisten lintas fold dan tail event.

Catatan integritas submission: pengguna telah mengonfirmasi bahwa skor RMSE 1,61079 berasal dari `submission_2.csv`, yang identik dengan `submission.csv` di handoff. `submission_1.csv` berasal dari model lain dan tidak dipakai dalam analisis maupun pemilihan V2.

## 13. Arsitektur V2 yang diimplementasikan

V2 mempertahankan model direct Huber sebagai guardrail dan menambahkan dua model residual musiman. Seluruh bobot ensemble dipilih hanya dari rolling-origin OOF, tanpa memakai skor leaderboard.

### 13.1 Quality mask target

Setiap observasi dibandingkan dengan dua tetangga valid sebelum dan sesudahnya, maksimum berjarak 45 hari. Ambang anomaly menggabungkan median absolute sequential difference dan robust scale per pos. Aturannya konservatif:

- reset nonpositif pada pos dengan elevasi median di atas 5 m dan isolated anomaly sangat ekstrem diberi bobot 0,05;
- isolated anomaly lain diberi bobot 0,20;
- observasi normal dan event tinggi yang berkelanjutan tetap berbobot 1,00.

Pada seluruh train, mask V2 menurunkan bobot 63 dari 84.396 label: 37 isolated anomaly, 22 very-extreme isolated, dan 4 nonpositive reset. Kasus Jurug 0 setelah gap Februari 2025 kini tertangkap. Label tidak pernah dihapus atau diimputasi.

### 13.2 Baseline musiman cross-fit

Baseline dibentuk per `(nama_pos, hour, day-of-year)` memakai kernel Gaussian melingkar dengan bandwidth 21 hari. Tahun kabisat dipetakan ke kalender 365 hari dan observasi lama diberi recency decay dengan half-life 1,5 tahun. Secara konseptual:

```text
B(s,h,d) = weighted_mean(TMA | station=s, hour=h, circular_distance(day,d))
weight    = quality_weight * Gaussian(distance / 21) * recency_half_life(1.5 year)
```

Untuk target training, baseline dihitung 5-fold cross-fit berdasarkan tanggal; seluruh baris pada tanggal holdout dikeluarkan dari lookup baseline. Untuk validation dan test, lookup hanya memakai histori sebelum cutoff. Dengan demikian tidak ada target baris sendiri atau label masa depan yang masuk ke baseline.

### 13.3 Tiga komponen model

1. **Direct Huber anchor** memprediksi target robust-normalized per pos `(TMA - median_pos) / scale_pos`, dengan `quality_weight`.
2. **Residual Huber** memprediksi `(TMA - baseline_musiman) / scale_pos`, dengan `quality_weight`. Baseline juga diberikan sebagai fitur numerik.
3. **Residual RMSE** memakai target residual yang sama, tetapi sample weight `quality_weight * scale_pos^2`. Faktor `scale_pos^2` menyelaraskan squared error pada target normalized dengan squared error dalam meter.

Setiap komponen final memakai ensemble seed 17, 41, dan 83. Prediksi akhir adalah:

```text
residual_ensemble = (1 - alpha) * residual_huber + alpha * residual_rmse
prediction        = (1 - beta)  * direct_anchor  + beta  * residual_ensemble
```

Grid joint 11 x 11 mencari `alpha` dan `beta` dengan objective `mean(fold quality-RMSE) + 0.1 * std(fold quality-RMSE)`. Penalti simpangan fold mengurangi kecenderungan memilih kombinasi yang hanya unggul pada satu regime. Handoff leaderboard `handoff_20260715T065403Z_all` memilih `alpha=0,2` dan `beta=0,5`, atau bobot efektif 50% direct Huber, 40% residual Huber, dan 10% residual RMSE.

Median best iteration yang dipakai final fit leaderboard adalah 323 untuk anchor, 737 untuk residual Huber, dan 657 untuk residual RMSE. Iterasi dipisahkan karena kurva early stopping ketiga objective berbeda jauh.

### 13.4 Hasil rolling-origin V2

| Fold | V1 direct Huber | V2 anchor | Residual Huber | Residual RMSE | Blend V2 |
|---|---:|---:|---:|---:|---:|
| sep_2023 | 2,6988 | 2,6236 | 2,5631 | 2,5603 | **2,5756** |
| may_2024 | 0,8605 | 0,8445 | 0,8328 | 0,8735 | **0,7651** |
| sep_2024 | **1,2267** | 1,2549 | 1,2848 | 1,3014 | 1,2398 |
| jan_2025 | **1,1906** | 1,1901 | 1,2349 | 1,2364 | 1,1944 |

Pooled raw OOF RMSE membaik dari 1,66058 menjadi 1,60029. Quality-weighted pooled OOF RMSE V2 adalah 0,96730. V2 membaik jelas pada stress fold `sep_2023` dan `may_2024`, tetapi sedikit turun pada `sep_2024` (+0,0131 m) dan `jan_2025` (+0,0038 m). Direct anchor tetap mendapat bobot 50% sebagai guardrail regime shift. Skor leaderboard V2 adalah RMSE 1,60422, hanya 0,00393 m dari pooled raw OOF.

Perbaikan terbesar tetap berasal dari Wonogiri Dam. Error pooled V2 terkonsentrasi pada Bojonegoro - Kali Kethek (38,2% squared error), Kedungupit (11,6%), Peren (8,5%), Wonogiri Dam (8,4%), dan Jurug (7,5%). Ini menjadi guardrail untuk iterasi berikutnya, bukan alasan melakukan tuning khusus pada leaderboard.

### 13.5 Artefak handoff V2

Selain file handoff V1, run V2 menyimpan:

- `v2_tuning.json`: bobot blend, iterasi setiap komponen, dan aturan seleksi;
- `metrics/v2_oof_predictions.csv`: anchor, kedua residual prediction, blend akhir, quality reason/weight, dan error;
- `metrics/v2_validation_metrics.csv`: raw dan quality-weighted metric setiap fold/komponen;
- `metrics/v2_blend_search.csv`: seluruh kandidat grid joint OOF;
- `metrics/v2_fold_station_metrics.csv`: RMSE, MAE, dan bias per fold-pos;
- `metrics/v2_feature_importance.csv` dan `v2_final_feature_importance.csv`: importance per komponen, fold, dan seed;
- `data/v2_quality_diagnostics.csv`: label yang diturunkan bobotnya beserta alasan;
- `data/v2_seasonal_lookup.csv`: lookup baseline final per pos/jam/hari musim;
- `models/v2_anchor_seed_<seed>.cbm`, `v2_huber_seed_<seed>.cbm`, dan `v2_rmse_seed_<seed>.cbm`;
- `v2_model_metadata.json` dan seluruh `training_progress/v2_*`.

Full final fit leaderboard telah dijalankan di Kaggle dalam run `handoff_20260715T065403Z_all` dan menghasilkan sembilan model final serta 21.780 prediksi finite. Karena manifest legacy run tersebut belum merekam hash notebook dan ringkasan konfigurasi efektif, koreksi historis disimpan tanpa mengubah ZIP asli di `handoff/handoff_20260715T065403Z_all_manifest_correction.json`, terikat pada SHA-256 arsip.

Mulai schema handoff 1.1, setiap run baru menulis langsung ke `manifest.json`:

- `notebook_sha256` dan `code_provenance`;
- `effective_config`, termasuk backend/accelerator aktual;
- `effective_model.blend.effective_component_weights`;
- `effective_model.iterations` untuk anchor, residual Huber, dan residual RMSE;
- `effective_model.tuning_source` dan selection rule;
- `artifact_sha256` untuk konfigurasi, tuning, metadata model, feature list, dan submission yang tersedia.

Jika file notebook tersedia pada filesystem runtime, manifest menyimpan raw file SHA-256 dan canonical source SHA-256 serta memverifikasi source hash terhadap fingerprint yang dideklarasikan di notebook. Jika file tidak tersedia, manifest memakai canonical source SHA-256 dengan status `declared_source_hash_only`; `notebook_hash_kind` membedakan hash canonical dari raw file hash. Untuk provenance paling kuat di Kaggle, `NOTEBOOK_PATH` tetap sebaiknya menunjuk ke salinan `.ipynb` yang benar-benar dieksekusi.

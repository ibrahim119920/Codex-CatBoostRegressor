# Log Pengembangan

## 2026-07-15 - Audit dan implementasi awal

Status akhir sesi: kode selesai ditulis dan diverifikasi secara statis/audit-only. Training model belum dijalankan.

### Aktivitas

- Menginventarisasi seluruh file proyek. `architecture.md` dan `logs.md` semula kosong; pipeline baru memakai sumber data resmi di bawah `datasets/` dan tidak mengubah notebook yang sudah ada.
- Membaca teks dan merender visual 7 halaman `HydroRIVERS_TechDoc_v10.pdf`. Halaman atribut dan proyeksi diperiksa secara visual.
- Membaca shapefile HydroRIVERS: 836.472 reach, CRS EPSG:4326, 15 kolom termasuk geometry.
- Memeriksa schema, coverage, key duplikat, missing value, distribusi, dan anomali train/test/lingkungan/koordinat.
- Mencocokkan 30 koordinat pos ke HydroRIVERS di UTM 49S dan memverifikasi relasi upstream-downstream dari `NEXT_DOWN`.
- Menulis pipeline training, validation, final ensemble, dan pembuatan submission; implementasi awal berada di `train.py` sebelum diminta dipindahkan ke notebook.
- Menulis dependensi runtime di `requirements.txt` dan keputusan desain lengkap di `architecture.md`.
- Menjalankan `python -m py_compile train.py`: berhasil.
- Menjalankan `python train.py --mode audit`: berhasil. Perintah ini tidak mengimpor CatBoost dan tidak melakukan fitting.
- Menjalankan builder preprocessing/feature engineering secara langsung tanpa fitting: berhasil membentuk 98 fitur model, 84.396 baris train, dan 21.780 baris test. Tidak ditemukan key duplikat, infinity, atau target timestamp tanpa pasangan hujan.
- Memverifikasi matriks upstream: 180 edge berbobot termasuk self-link dan jumlah bobot setiap target pos sama dengan 1 (dalam toleransi floating point).

### Temuan data

- Train: 84.396 baris, 30 pos, 1 Januari 2023 06.00 sampai 18 September 2025 18.00.
- Test/submission: 21.780 baris, 19 September 2025 06.00 sampai 18 Mei 2026 18.00; ID dan urutannya identik.
- Lingkungan: 888.480 baris atau 29.616 jam x 30 pos; tidak ada key yang hilang atau duplikat.
- Grid target teoritis memiliki 89.280 baris; 4.884 observasi target tidak tercatat. Missing paling umum terjadi pada 4-28 Februari 2025. Gunungsari tidak memiliki data sebelum 13 Agustus 2024.
- Tidak ada `NaN` eksplisit pada `tma_mdpl`, tetapi ada beberapa spike satu-titik yang sangat besar dan segera kembali ke baseline.
- Detektor spike konservatif menandai 37 dari 84.396 label (0,044%) untuk bobot 0,05; semua label tetap berada di dataset.
- Missing eksplisit lingkungan berjumlah 20.880 sel: 720 per variabel pada 18 Mei 2026 untuk soil/pressure/MJO dan 12.960 pada `nino_34` untuk 1-18 Mei 2026.
- Selain `NaN`, `solar_radiation_mj_m2` memuat 100.080 sentinel `-999`. Sentinel tersebut menyebabkan mean masa prediksi tampak negatif dan harus diubah menjadi missing.
- `rainfall_mm` dan `rainfall_openmeteo_mm` identik 100%.
- Nearest-river distance berada pada sekitar 2,7 m sampai 1.237 m; jarak disimpan sebagai fitur agar match yang lemah dapat dikenali model.

### Keputusan model

- Ensemble 3 seed CatBoost global, loss Huber, target robust-normalized per pos.
- Fitur eksogen non-rekursif agar validasi dan prediksi 242 hari mempunyai informasi yang sama.
- Rolling-origin validation dengan empat horizon panjang dan baseline median pos-bulan-jam.
- Missing label tidak diimputasi; isolated spike hanya diturunkan bobotnya, bukan dihapus.
- Agregat hujan/soil upstream dibangun dari traversal `NEXT_DOWN`, dengan decay jarak hidraulik.

### Verifikasi yang belum dijalankan

- CatBoost cross-validation, early stopping, dan final ensemble sengaja belum dijalankan sesuai instruksi.
- Karena model belum dilatih, belum ada angka MAE/RMSE, best iteration, feature importance, model `.cbm`, atau submission prediktif.

## 2026-07-15 - Pemindahan pipeline ke notebook

- Seluruh definisi dari `train.py` dipindahkan ke `notebook-150726.ipynb` dan dibagi menjadi 11 bagian yang dapat dibaca/diuji secara bertahap.
- Ditambahkan fungsi `run_pipeline()` sebagai pengganti CLI serta cell `MODE`, `DATA_DIR`, `ARTIFACTS_DIR`, dan `ITERATIONS`.
- Mode default notebook adalah `audit`; Run All tidak memanggil fitting CatBoost kecuali `MODE` diubah eksplisit menjadi `validate`, `fit`, atau `all`.
- `train.py` dihapus setelah pemindahan sehingga tidak ada dua implementasi yang dapat menyimpang.
- Notebook berhasil diparse sebagai nbformat 4.5 dengan 23 cell; seluruh 11 code cell lolos pemeriksaan AST Python.
- Seluruh cell kemudian diverifikasi dalam mode default `audit`: audit menghasilkan jumlah baris/pos yang sama seperti implementasi awal dan memastikan modul CatBoost tidak dimuat.
- Training tetap belum dijalankan.

## 2026-07-15 - Dukungan Kaggle dan handoff training

- Menambahkan deteksi otomatis input kompetisi di `/kaggle/input/competitions/sebelas-maret-statistics-data-science-2026` beserta seluruh isi `data_pendukung/`.
- Menetapkan `/kaggle/working/` sebagai output root pada Kaggle, dengan fallback lokal `datasets/` dan `artifacts/`.
- Menambahkan run ID bertimestamp UTC berbentuk `handoff_YYYYMMDDTHHMMSSZ_<mode>`.
- Menambahkan manifest, konfigurasi, versi runtime, inventory input, data audit, matriks fitur opsional, diagnostic sample weight, drift report, dan README handoff.
- Menambahkan log CatBoost per fold/seed, snapshot berkala 300 detik, OOF prediction, metrik per pos, residual slice, feature importance, model fold, dan model ensemble final.
- Menambahkan ZIP `in_progress` setelah feature preparation dan setiap fold/seed; run sukses menghasilkan ZIP `complete`, sedangkan exception menghasilkan ZIP `failed` beserta traceback.
- Memperbaiki export feature matrix agar nama key yang juga menjadi fitur kategorikal tidak menghasilkan kolom duplikat.
- Seluruh 12 code cell lolos AST. Notebook berhasil dijalankan dalam mode `audit` dengan path lokal fallback dan tanpa memuat CatBoost.
- Mekanisme handoff diuji memakai frame dummy: run ID waktu, Parquet, manifest, ZIP in-progress, penggantian menjadi ZIP complete, serta struktur anggota ZIP semuanya valid.
- Handoff pra-training juga diuji dengan seluruh 98 fitur nyata dan `SAVE_FEATURE_MATRICES=False`: diagnostic sample, target normalization, upstream weights, drift report, manifest, dan ZIP in-progress berhasil dibuat; hasil uji kemudian dibersihkan.
- Parameter `train_dir`, `allow_writing_files`, dan snapshot diverifikasi terhadap dokumentasi resmi CatBoost.
- Training CatBoost tetap belum dijalankan.

## 2026-07-15 - Analisis handoff run pertama

- Membuka dan memverifikasi `handoff_20260714T231248Z_all_complete.zip`: 78 file, status complete, 98 fitur, empat fold, tiga seed, dan final iteration 193.
- Skor leaderboard yang diberikan pengguna: RMSE 1,61079.
- Menghitung ulang pooled OOF: MAE 0,62263, RMSE 1,66058, bias +0,12224 m. Leaderboard hanya 0,04979 lebih rendah, sehingga pooled validation cukup merepresentasikan performa eksternal.
- Memuat tiga model final CatBoost 1.2.10 dan melakukan inference diagnostik pada train tanpa retraining. Raw train RMSE 2,23044; weighted train RMSE 0,87525; train RMSE tanpa 37 spike 0,73787.
- OOF RMSE tanpa spike yang sudah ditandai adalah 1,11480. Gap clean train-OOF sekitar 0,37693 m atau 51% relatif terhadap clean train.
- RMSE fold: sep_2023 2,69881; may_2024 0,86051; sep_2024 1,22669; jan_2025 1,19062. Fold sep_2024 diperlakukan sebagai horizon-aligned validation utama.
- Pada iterasi 193, validation MAE ternormalisasi hanya 0,0014-0,0162 lebih buruk dari best masing-masing fold; menambah tree saja bukan prioritas utama.
- 5% target tertinggi setelah pembersihan memiliki bias -1,50809 m dan RMSE 2,00759 m; 50% target terendah bias +0,45594 m. Didiagnosis regresi ke rata-rata akibat Huber/normalisasi dan baseline musiman yang belum eksplisit.
- Bojonegoro - Kali Kethek menyumbang 35,9% OOF squared error dan Wonogiri Dam 14,6%. Jurug 0 setelah gap Februari belum tertangkap quality mask dan memberi error sekitar 81 m pada dua fold.
- Standard deviation antar-seed pada test rata-rata hanya 0,03844 m; lebih banyak seed identik diperkirakan berkontribusi kecil.
- Arah iterasi kedua ditetapkan: ensemble Huber anchor + RMSE scale-weighted, quality mask lintas gap, station seasonal baseline-residual cross-fit, evaluasi tail/worst-station, lalu travel-time upstream dan feature ablation.
- Pengguna mengonfirmasi bahwa submission handoff/`submission_2.csv` memperoleh RMSE leaderboard 1,61079. `submission_1.csv` berasal dari model lain dan diabaikan.
- Tidak ada retraining atau perubahan model dilakukan pada tahap analisis ini.

## 2026-07-15 - Implementasi dan evaluasi V2

### Perubahan model

- Mengonfirmasi `submission_2.csv` sebagai hasil run pertama dengan leaderboard RMSE 1,61079; `submission_1.csv` tidak digunakan.
- Menambahkan quality diagnostics V2 berbasis empat nearest temporal neighbors sampai 45 hari, robust sequential-difference scale, dan robust station scale. Semua label dipertahankan; hanya bobotnya yang diubah.
- Menambahkan baseline Gaussian circular per pos, jam, dan day-of-year dengan bandwidth 21 hari serta recency half-life 1,5 tahun.
- Baseline target training dihitung 5-fold cross-fit berdasarkan tanggal untuk mencegah self-target leakage. Lookup validation/test hanya memakai bagian training sebelum cutoff.
- Menambahkan direct Huber anchor, residual Huber, dan residual RMSE. Residual RMSE memakai bobot `quality_weight * station_scale^2` agar objective normalized sejalan dengan RMSE meter.
- Menambahkan joint OOF grid untuk memilih fraksi residual RMSE dan fraksi keseluruhan residual terhadap direct anchor. Selection score adalah mean quality-weighted RMSE antar-fold ditambah 0,1 kali standard deviation antar-fold.
- Orkestrator `run_pipeline()` sekarang menjalankan V2 untuk mode `validate`, `fit`, dan `all`; fungsi V1 dipertahankan di notebook hanya untuk reproducibility.
- Menambahkan artefak V2: tuning, OOF tiga komponen, blend search, station metrics, feature importance per komponen, quality diagnostics, seasonal lookup, sembilan model final, metadata, dan training progress.

### Pemeriksaan kode dan data

- Notebook valid sebagai JSON dan seluruh code cell berhasil dikompilasi.
- Full feature smoke test membentuk 84.396 baris train, 21.780 test, dan 98 fitur tanpa baseline; baseline V2 menjadi fitur ke-99 saat fitting residual.
- Seasonal lookup membentuk 32.850 kombinasi pos-jam-hari dan menghasilkan baseline finite pada seluruh sampel uji.
- Quality mask menurunkan bobot 63 label: 37 `isolated_anomaly`, 22 `very_extreme_isolated`, dan 4 `nonpositive_reset`. Jurug 0 pada 28 Februari 2025 kini tertangkap.

### Rolling-origin validation

Delapan residual model dan empat direct-anchor model dilatih dengan cutoff yang sama seperti V1. Hasil RMSE meter:

| Fold | V1 | Anchor V2 | Residual Huber | Residual RMSE | Blend V2 |
|---|---:|---:|---:|---:|---:|
| sep_2023 | 2,69881 | 2,62361 | 2,56309 | 2,56033 | 2,57560 |
| may_2024 | 0,86051 | 0,84452 | 0,83278 | 0,87350 | 0,76515 |
| sep_2024 | 1,22669 | 1,25491 | 1,28475 | 1,30140 | 1,23982 |
| jan_2025 | 1,19062 | 1,19011 | 1,23489 | 1,23638 | 1,19440 |

- Handoff produksi memilih `blend_rmse_fraction=0,2` dan `blend_residual_fraction=0,5`, sehingga bobot efektif final adalah 50% anchor, 40% residual Huber, dan 10% residual RMSE.
- Median iteration produksi: anchor 323, residual Huber 737, residual RMSE 657.
- Pooled raw OOF RMSE turun dari 1,66058 (V1) menjadi 1,60029 (V2), perbaikan 0,06029 m atau 3,63%.
- Quality-weighted pooled OOF RMSE V2 adalah 0,96730.
- V2 unggul pada dua fold dan sedikit kalah pada dua fold terbaru: `sep_2024` +0,01313 m dan `jan_2025` +0,00378 m. Direct anchor 50% dipertahankan sebagai guardrail regime shift.
- Skor leaderboard V2 adalah RMSE 1,60422, hanya 0,00393 m di atas pooled raw OOF.
- Lima sumber squared error V2 terbesar adalah Bojonegoro - Kali Kethek, Kedungupit, Peren, Wonogiri Dam, dan Jurug.

### Integration test final fit

- Final-fit path diuji dengan satu seed dan dua iterasi untuk tiap komponen, bukan sebagai model produksi.
- Uji menghasilkan tiga model `.cbm`, lookup musiman, quality diagnostics, metadata, dan submission 21.780 baris tanpa missing/non-finite prediction.
- Full final training dan inference tidak dijalankan lokal. Run produksi kemudian selesai di Kaggle sebagai `handoff_20260715T065403Z_all` dan memperoleh leaderboard RMSE 1,60422.

## 2026-07-17 - Sinkronisasi dokumentasi dan provenance handoff

- Mengaudit ulang `handoff_20260715T065403Z_all_complete.zip` dan menetapkan `config.json`, `v2_tuning.json`, metric CSV, dan submission di dalam ZIP sebagai sumber kebenaran untuk model leaderboard.
- Mengoreksi bobot blend, iteration count, fold metric, pooled OOF, dan status final fit pada `architecture.md` serta bagian V2 log ini.
- Menaikkan schema handoff dari 1.0 menjadi 1.1.
- Menambahkan SHA-256 notebook ke manifest melalui `notebook_sha256`, jenis hash melalui `notebook_hash_kind`, serta raw/canonical hash, path, status, dan ukuran melalui `code_provenance`. Canonical source hash tetap tersedia saat file `.ipynb` tidak terlihat di runtime Kaggle.
- Menambahkan `effective_config` yang mencatat parameter model, fold, mode, requested iteration, feature-matrix toggle, dan accelerator aktual.
- Menambahkan `effective_model` yang mengekspansi `alpha/beta` menjadi bobot efektif anchor/residual Huber/residual RMSE, iteration count tiap komponen, seed, selection rule, dan sumber tuning.
- Menambahkan `artifact_sha256` untuk file konfigurasi, feature list, tuning, metadata model, dan submission yang tersedia sebelum ZIP complete dibuat.
- Menjaga ZIP leaderboard lama immutable. Koreksi historis dibuat sebagai `handoff/handoff_20260715T065403Z_all_manifest_correction.json`, terikat ke SHA-256 ZIP, manifest legacy, config, tuning, dan submission.
- Hash notebook asli run legacy tidak dapat direkonstruksi dengan aman; sidecar menandainya `legacy_missing` alih-alih mengaitkan hash notebook saat ini secara keliru.

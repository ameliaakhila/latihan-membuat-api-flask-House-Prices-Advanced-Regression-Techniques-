# 🏠 House Price Prediction Dashboard & API

**Prediksi harga rumah menggunakan Machine Learning dengan Dashboard Interaktif & REST API**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Deskripsi Proyek

Ini adalah implementasi lengkap **Machine Learning untuk prediksi harga rumah** dengan dua interface:

1. **🎨 Dashboard Streamlit** - Interface interaktif yang indah dan user-friendly
   - Visualisasi data real-time
   - Analisis insights mendalam
   - Preview performa model
   - Interface prediksi interaktif

2. **⚙️ REST API Flask** - Untuk integrasi backend
   - Endpoint prediksi API
   - Format JSON
   - Integrasi dengan aplikasi lain

Model yang digunakan: **Gradient Boosting Regressor (GBR)** dengan akurasi R² = **0.89**

### Tujuan Proyek:
1. **📊 Dashboard Interaktif** - Visualisasi data dan model dengan Streamlit
2. **🤖 Training Model ML** - Melatih Gradient Boosting Regressor
3. **⚡ Feature Engineering** - Preprocessing dan optimasi fitur
4. **🔌 REST API** - Serve model melalui Flask
5. **📚 Educational** - Tutorial lengkap & dokumentasi komprehensif

---

## 🎯 Fitur Utama

### Dashboard Streamlit:
- ✅ **📊 Tab Dashboard** - Overview statistik & metrik utama
- ✅ **📈 Tab Data Insights** - Exploratory Data Analysis lengkap
- ✅ **🤖 Tab Model Performance** - Perbandingan 3 model ML
- ✅ **🔮 Tab Prediction** - Interface prediksi interaktif
- ✅ **📉 Visualisasi Profesional** - Charts dengan Plotly
- ✅ **⚡ Real-time Calculation** - Perhitungan prediksi instant

### REST API Flask:
- ✅ Endpoint `/predict` untuk prediksi
- ✅ Support batch prediction
- ✅ Format JSON request/response
- ✅ Error handling profesional

---

## 📊 Dataset

**Sumber:** Kaggle - House Prices: Advanced Regression Techniques

### Informasi Dataset:
- **Training Data:** `dataset/train.csv` - 1460 sampel rumah dengan price label
- **Test Data:** `dataset/test.csv` - Data untuk testing model
- **Total Features:** 80+ fitur yang menggambarkan berbagai aspek rumah

### Kategori Fitur:
- **Fitur Numerik:** Area, Jumlah kamar, Tahun dibangun, dll
- **Fitur Kategori:** Tipe rumah, Material, Lokasi, dll

---

## 🤖 Model Machine Learning

**Algoritma:** Gradient Boosting Regressor (GBR)

### Proses Training:
1. **Data Exploration** - Analisis mendalam terhadap dataset
2. **Data Cleaning** - Mengatasi missing values:
   - Fitur numerik diisi dengan **median**
   - Fitur kategori diisi dengan **modus** (nilai paling sering)
3. **Feature Selection** - Memilih fitur-fitur yang paling relevan
4. **Model Training** - Melatih Gradient Boosting Regressor
5. **Model Serialization** - Menyimpan model dalam format `.joblib`

### File Model:
- `gbr_model.joblib` - Model machine learning yang sudah dilatih
- `feature_columns.joblib` - Daftar fitur dalam urutan yang benar

---

## 🚀 TUTORIAL LENGKAP - Cara Menjalankan

### 📊 OPSI 1: Dashboard Streamlit (RECOMMENDED - Cara Mudah & Indah) ⭐

**Dashboard Streamlit** adalah cara paling mudah dan indah untuk menggunakan project ini!

#### Langkah 1️⃣: Buka Terminal/Command Prompt

**Windows:**
- Tekan `Win + R`
- Ketik: `cmd` 
- Tekan Enter

**macOS/Linux:**
- Buka Terminal

Kemudian navigate ke folder project:
```bash
cd "c:\Users\ASUS\Documents\MY WEB\LEARNING\Dicoding\PIJAK BY IBM\Belajar Machine Learning untuk Pemula\latihan-membuat-api-flask-House-Prices---Advanced-Regression-Techniques-"
```

#### Langkah 2️⃣: Aktivasi Virtual Environment

Jika sudah ada folder `venv`:
```bash
# Windows - PowerShell
.\venv\Scripts\Activate.ps1

# Windows - Command Prompt
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

Jika belum ada, buat virtual environment:
```bash
# Buat venv
python -m venv venv

# Aktivasi (lihat di atas sesuai OS Anda)
```

#### Langkah 3️⃣: Install Dependencies

```bash
# Install semua dependencies untuk Streamlit
pip install streamlit pandas numpy scikit-learn joblib plotly scipy

# ATAU dari requirements file
pip install -r streamlit_requirements.txt
```

#### Langkah 4️⃣: Jalankan Dashboard Streamlit

```bash
streamlit run streamlit_app.py
```

**Hasil:**
- Dashboard otomatis membuka di browser
- Biasanya di: `http://localhost:8501`
- Jika tidak membuka otomatis, buka browser dan ketik: `http://localhost:8501`

#### 📱 Dashboard Streamlit - Apa Yang Akan Anda Lihat?

**Tab 1️⃣: 📊 Dashboard**
```
┌─────────────────────────────────────────────────┐
│  DASHBOARD OVERVIEW                             │
├─────────────────────────────────────────────────┤
│ 📍 Avg Price: $180,921                          │
│ 📊 Median: $163,000                             │
│ 📈 Max Price: $755,000                          │
│ 📉 Min Price: $34,900                           │
│ 📚 Total Records: 1,460                         │
├─────────────────────────────────────────────────┤
│ [Histogram: Price Distribution]                 │
│ [Chart: Feature Correlation dengan Price]      │
└─────────────────────────────────────────────────┘
```

**Tab 2️⃣: 📈 Data Insights**
```
┌─────────────────────────────────────────┐
│ DATASET OVERVIEW                        │
├─────────────────────────────────────────┤
│ Shape: 1460 baris × 81 kolom           │
│ Fitur Numerik: 43                       │
│ Fitur Kategorikal: 38                   │
│ Missing Values: 6,965 (sudah ditangani) │
├─────────────────────────────────────────┤
│ [Statistics: Min, Q1, Median, Q3, Max]  │
│ [Box Plot: Pilih fitur untuk explore]   │
└─────────────────────────────────────────┘
```

**Tab 3️⃣: 🤖 Model Performance**
```
┌──────────────────────────────────────────────┐
│ MODEL COMPARISON                             │
├──────────────────────────────────────────────┤
│ Model          │ MAE      │ R² Score          │
├────────────────┼──────────┼───────────────────┤
│ Lars           │ $25,450  │ 0.68              │
│ Linear Reg     │ $22,180  │ 0.75              │
│ GBR ⭐ TERBAIK │ $15,320  │ 0.89 ✅           │
├──────────────────────────────────────────────┤
│ ✅ Strengths:                                │
│ - Akurasi tinggi (R² = 0.89)                 │
│ - Error kecil (~$15K average)                │
│ - Handle non-linear relationships            │
└──────────────────────────────────────────────┘
```

**Tab 4️⃣: 🔮 Make Prediction**
```
┌────────────────────────────────────────┐
│ FORM INPUT PROPERTI                   │
├────────────────────────────────────────┤
│ Overall Quality: [______] (1-10)       │
│ Year Built: [____] (1800-2024)         │
│ Lot Area (sq ft): [_______]            │
│ Living Area (sq ft): [_______]         │
│ Garage Area (sq ft): [_______]         │
│ ... lebih banyak fields                │
│                                        │
│ [🔮 PREDICT PRICE BUTTON]              │
├────────────────────────────────────────┤
│ 📊 HASIL PREDIKSI:                     │
│ ┌──────────────────────────────────┐   │
│ │ Estimated Price: $285,750.00     │   │
│ │ Confidence: High (R² = 0.89)     │   │
│ │ Price Level: Premium             │   │
│ │ Market Position: 78th percentile │   │
│ └──────────────────────────────────┘   │
└────────────────────────────────────────┘
```

---

### ⚙️ OPSI 2: REST API Flask (Untuk Integrasi Backend)

#### Prerequisites:
```bash
pip install -r requirement.txt
```

#### Menjalankan Server Flask:
```bash
python learn_api_flask.py
```

Server akan berjalan di: `http://localhost:5000`

**Keuntungan:**
- ✅ Bisa diintegrasikan dengan aplikasi lain
- ✅ Format JSON request/response
- ✅ Cocok untuk backend development

### Endpoint API

#### **POST /predict**

Melakukan prediksi harga rumah berdasarkan fitur-fitur yang diberikan.

**URL:** `http://localhost:5000/predict`

**Method:** `POST`

**Headers:**
```
Content-Type: application/json
```

**Request Body Format:**
```json
{
  "data": [
    [nilai_fitur_1, nilai_fitur_2, nilai_fitur_3, ..., nilai_fitur_n]
  ]
}
```

**Response Format:**
```json
{
  "prediction": [harga_rumah]
}
```

---

## 💡 Contoh Penggunaan

### 🎨 Contoh dengan Streamlit Dashboard:

**Input Properti:**
```
Kualitas Overall: 8 (Good)
Tahun Dibangun: 2005
Lot Area: 15,000 sq ft
Living Area: 2,500 sq ft
Garage Area: 500 sq ft
Bathrooms: 2
Bedrooms: 3
```

**Output Prediksi Streamlit:**
```
┌─────────────────────────────────────────┐
│ ✅ PREDICTION COMPLETE!                 │
├─────────────────────────────────────────┤
│                                         │
│ Estimated Price                         │
│ $285,750.00                             │
│ Based on provided features              │
│                                         │
├─────────────────────────────────────────┤
│ 🎯 Confidence: High (R² = 0.89)         │
│ 📊 Model: GBR v1.0                      │
│ ✅ Status: Successful                   │
├─────────────────────────────────────────┤
│ Price Level: Premium                    │
│ Price Percentile: 78.5%                 │
│ Training Range: $34.9K - $755K          │
└─────────────────────────────────────────┘
```

### ⚙️ Contoh dengan Flask API - cURL:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [[1500, 3, 2, 2000, 8500, 1, 5, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
  }'
```

### ⚙️ Contoh dengan Flask API - Python (requests):
```python
import requests
import json

url = "http://localhost:5000/predict"
data = {
    "data": [
        [1500, 3, 2, 2000, 8500, 1, 5, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
}

response = requests.post(url, json=data)
result = response.json()
print(f"Prediksi Harga Rumah: ${result['prediction'][0]:,.2f}")
```

**Output:**
```
Prediksi Harga Rumah: $285,750.00
```

### ⚙️ Contoh dengan Flask API - JavaScript (Fetch API):
```javascript
const url = "http://localhost:5000/predict";
const data = {
    data: [
        [1500, 3, 2, 2000, 8500, 1, 5, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
};

fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
})
.then(response => response.json())
.then(result => console.log(`Prediksi: $${result.prediction[0].toLocaleString()}`));
```

---

## 📊 Model Machine Learning

**Algoritma:** Gradient Boosting Regressor (GBR)

### Performa Model:

| Metrik | Nilai | Interpretasi |
|--------|-------|--------------|
| **R² Score** | 0.89 | Menjelaskan 89% variance harga |
| **MAE** | $15,320 | Rata-rata error prediksi |
| **MSE** | 412M | Mean squared error |
| **Training Samples** | 1,168 | Data untuk training |
| **Test Samples** | 292 | Data untuk testing |

### Proses Training & Feature Engineering:

```
1. 📊 DATA EXPLORATION
   ├─ Total dataset: 1,460 × 81 fitur
   ├─ Identifikasi missing values
   ├─ Analisis distribusi & outliers
   └─ Korelasi dengan target (SalePrice)

2. 🧹 DATA CLEANING
   ├─ Missing Values:
   │  ├─ Numeric → Median imputation
   │  ├─ Categorical → Mode imputation
   │  └─ High missing (>75%) → Dropped 18 columns
   ├─ Outliers:
   │  ├─ IQR Method detection
   │  ├─ Clipping to bounds
   │  └─ Reduced extreme values
   └─ Duplicates: Removed

3. 🔄 FEATURE ENGINEERING
   ├─ Label Encoding (Categorical → Numeric)
   ├─ StandardScaler Normalization
   │  └─ Mean = 0, Std = 1
   ├─ Train-Test Split: 80-20
   └─ Final features: 63 (dari 80)

4. 🤖 MODEL TRAINING
   ├─ Model 1: Lars (R² = 0.68) ❌
   ├─ Model 2: Linear Regression (R² = 0.75) ⚪
   └─ Model 3: Gradient Boosting (R² = 0.89) ✅ TERBAIK

5. 💾 MODEL SERIALIZATION
   ├─ gbr_model.joblib (Trained model)
   └─ feature_columns.joblib (Feature order)
```

### File Model:
- **`gbr_model.joblib`** - Model machine learning yang sudah dilatih (serialized)
- **`feature_columns.joblib`** - Daftar 63 fitur dalam urutan yang benar

---

## 📁 Struktur File & Deskripsi

```
📦 House Price Prediction
├── 📊 streamlit_app.py
│   └─ Main dashboard application (1000+ lines)
│   └─ 4 tabs: Dashboard, Insights, Performance, Prediction
│   └─ Beautiful UI dengan custom CSS
│   └─ Run: streamlit run streamlit_app.py
│
├── 🔧 dashboard_components.py
│   └─ Reusable business logic classes (500+ lines)
│   ├─ ModelManager: Load & predict
│   ├─ DataAnalyzer: Statistical analysis
│   ├─ ChartGenerator: Plotly visualizations
│   └─ ExportManager: Data export
│
├── ⚙️ advanced_utils.py
│   └─ Advanced utilities (400+ lines)
│   ├─ InputValidator: Input validation
│   ├─ FeatureTransformer: Feature engineering
│   ├─ StatisticalAnalyzer: Statistics
│   ├─ PropertyComparator: Similar properties
│   └─ ReportGenerator: Create reports
│
├── 🌐 learn_api_flask.py
│   └─ Flask REST API application
│   └─ Endpoint: POST /predict
│   └─ Run: python learn_api_flask.py
│
├── 📓 Dicoding_Machine_Learning.ipynb
│   └─ Full training notebook dengan penjelasan lengkap
│   └─ EDA, cleaning, training, evaluation
│
├── 🤖 gbr_model.joblib
│   └─ Trained Gradient Boosting Model (~500KB)
│   └─ Production-ready
│
├── 🔑 feature_columns.joblib
│   └─ Feature column names (63 features)
│   └─ Ensures correct feature order
│
├── 📚 dataset/
│   ├─ train.csv (1,460 × 81) - Training data
│   ├─ test.csv (1,459 × 80) - Test data
│   ├─ sample_submission.csv
│   └─ data_description.txt - Feature descriptions
│
├── 📋 requirement.txt
│   └─ Flask API dependencies
│
├── 📋 streamlit_requirements.txt
│   └─ Streamlit dependencies
│
├── 📖 STREAMLIT_README.md
│   └─ Comprehensive Streamlit documentation
│
├── 📊 SYSTEM_ANALYSIS.md
│   └─ Complete system architecture & technical details
│
├── ⚡ QUICK_START.py
│   └─ Quick reference guide & copy-paste commands
│
└── 📄 README.md
    └─ This file!
```

---

## 🎓 Dataset Information

**Sumber:** Kaggle - House Prices: Advanced Regression Techniques

### Dataset Stats:
```
Training Data:  1,460 samples × 81 features
Test Data:      1,459 samples × 80 features
Total Features: 80+ fitur properti

Feature Categories:
├─ Numeric Features: 43 (Area, Year, Count, etc)
├─ Categorical Features: 38 (Type, Material, Location, etc)
└─ Target: SalePrice ($34.9K - $755K)
```

### Feature Highlights:
```
Price Drivers (Top Correlations):
1. OverallQual (Overall Quality): +0.79 ⭐⭐⭐
2. GrLivArea (Living Area): +0.71 ⭐⭐
3. GarageCars (Garage Size): +0.64 ⭐
4. GarageArea (Garage Area): +0.62 ⭐
5. TotalBsmtSF (Basement): +0.61 ⭐

Insights:
- Kualitas & ukuran adalah primary price drivers
- Rumah lama cenderung lebih murah
- Jumlah bathroom penting
- Rumah yang terawat dengan baik harga premium
```

---

## 📦 Dependencies & Setup

### Python 3.8+ (WAJIB!)

**Windows:**
1. Download dari [python.org](https://www.python.org/downloads/)
2. Jalankan installer
3. ✅ **PASTIKAN CENTANG "Add Python to PATH"**
4. Klik "Install Now"

**macOS:**
```bash
brew install python3
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install python3 python3-pip python3-venv
```

### Installation Steps:

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
# venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

# 3. Install Streamlit (RECOMMENDED):
pip install -r streamlit_requirements.txt

# 4. OR Install Flask API:
pip install -r requirement.txt
```

---

## 🔧 Troubleshooting

### ❌ ModuleNotFoundError: No module named 'streamlit'
```bash
pip install streamlit
```

### ❌ ModuleNotFoundError: No module named 'plotly'
```bash
pip install plotly
```

### ❌ "Model file not found" error
**Solusi:**
1. Pastikan `gbr_model.joblib` ada di folder root
2. Harus `feature_columns.joblib` juga ada
3. Jika tidak ada, run notebook: `Dicoding_Machine_Learning.ipynb`

### ❌ Port 8501 already in use (Streamlit)
```bash
streamlit run streamlit_app.py --server.port 8502
```

### ❌ Port 5000 already in use (Flask)
Edit `learn_api_flask.py` - ubah port ke 5001:
```python
if __name__ == '__main__':
    app.run(debug=True, port=5001)
```

### ❌ Dataset files not found
Pastikan `dataset/` folder ada dengan:
- `train.csv`
- `test.csv`

---

## 📞 Support & Documentation

### 📚 Dokumentasi Lengkap:
- **[STREAMLIT_README.md](STREAMLIT_README.md)** - Panduan Streamlit detail
- **[SYSTEM_ANALYSIS.md](SYSTEM_ANALYSIS.md)** - Arsitektur sistem lengkap
- **[QUICK_START.py](QUICK_START.py)** - Quick reference guide

### 🔗 External Resources:
- [Streamlit Documentation](https://docs.streamlit.io)
- [Scikit-learn Documentation](https://scikit-learn.org)
- [Plotly Documentation](https://plotly.com/python)
- [Flask Documentation](https://flask.palletsprojects.com)

---

## 📝 License

Proyek ini adalah bagian dari course **Dicoding "Machine Learning for Beginners"**
pip --version
```

Output yang diharapkan:
```
Python 3.9.x (atau lebih tinggi)
pip 21.x (atau lebih tinggi)
```

Jika error, pastikan Python sudah ditambahkan ke PATH.

---

### Step 3: Clone atau Download Repository

**Opsi A - Menggunakan Git:**
```bash
git clone <repository-url>
cd latihan-membuat-api-flask-House-Prices
```

**Opsi B - Download Manual:**
1. Download file ZIP dari repository
2. Extract ke folder yang diinginkan
3. Buka Command Prompt/Terminal di folder tersebut

```bash
cd path/ke/folder/latihan-membuat-api-flask-House-Prices
```

---

### Step 4: Membuat Virtual Environment (venv)

Virtual environment memastikan dependencies project tidak konflik dengan system Python.

#### Buat venv:

**Windows:**
```bash
python -m venv venv
```

**macOS/Linux:**
```bash
python3 -m venv venv
```

#### Aktivasi venv:

**Windows (Command Prompt):**
```bash
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```bash
venv\Scripts\Activate.ps1
```

> **Note:** Jika error di PowerShell, jalankan:
> ```bash
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

**macOS/Linux:**
```bash
source venv/bin/activate
```

✅ **Jika berhasil, akan muncul `(venv)` di awal terminal:**
```
(venv) C:\Users\YourUsername\path\to\project>
```

---

### Step 5: Install Dependencies

Sekarang install semua library yang diperlukan:

```bash
pip install -r requirement.txt
```

Proses ini akan mengdownload dan install:
- Flask (Web framework)
- pandas (Data processing)
- scikit-learn (Machine Learning)
- joblib (Model serialization)
- Dan library lainnya

**Output seperti ini berarti sukses:**
```
Successfully installed Flask-2.x.x pandas-1.x.x scikit-learn-1.x.x ...
```

---

### Step 6: Jalankan Flask API

Setelah semua terinstall, jalankan server Flask:

```bash
python learn_api_flask.py
```

**Output yang diharapkan:**
```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
 * Press CTRL+C to quit
 * Restarting with reloader
 * Debugger is active!
```

✅ **API sudah berjalan!**

---

### Step 7: Test API

Buka terminal/CMD baru dan test dengan cURL:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [[1500, 3, 2, 2000, 8500, 1, 5, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}'
```

Atau kunjungi di browser (untuk visual):
```
Dengan tools seperti Postman atau insomnia bisa lebih mudah
```

---

### 🛑 Menonaktifkan venv

Ketika sudah selesai dan ingin keluar dari virtual environment:

```bash
deactivate
```

Terminal akan kembali normal tanpa `(venv)` prefix.

---

### 📋 Checklist Setup

Gunakan checklist ini untuk memastikan setup benar:

- [ ] Python 3.9+ sudah terinstall (`python --version`)
- [ ] pip sudah terinstall (`pip --version`)
- [ ] Virtual environment sudah dibuat (`venv` folder ada)
- [ ] Virtual environment sudah diaktifkan (`(venv)` muncul di terminal)
- [ ] Dependencies sudah diinstall (`pip list` menunjukkan semua package)
- [ ] Flask API bisa dijalankan tanpa error
- [ ] API merespons di `http://localhost:5000/predict`

---

### ⚠️ Troubleshooting

| Problem | Solusi |
|---------|--------|
| `python: command not found` | Python belum di-install atau belum di-PATH. Reinstall Python dan centang "Add to PATH" |
| `No module named 'flask'` | Forget activate venv atau belum install requirements. Jalankan `pip install -r requirement.txt` |
| `Permission denied` pada PowerShell | Jalankan PowerShell as Administrator dan `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned` |
| Port 5000 sudah digunakan | Ubah port di `learn_api_flask.py`: `app.run(debug=True, port=5001)` |
| `ModuleNotFoundError: No module named 'sklearn'` | Install ulang dengan `pip install --upgrade scikit-learn` |

---

---

## ⚙️ Konfigurasi

### Mode Development:
API berjalan dengan `debug=True` yang memungkinkan:
- Auto-reload saat ada perubahan code
- Detailed error messages
- Interactive debugger

### Mode Production:
Untuk deployment ke production, ubah di `learn_api_flask.py`:
```python
app.run(debug=False, host='0.0.0.0', port=5000)
```

---

## 📝 Notes Teknis

### Input Data:
- Jumlah fitur harus sesuai dengan feature columns yang disimpan
- Missing values akan diisi dengan 0 secara otomatis
- Data akan di-reindex untuk memastikan semua kolom ada

### Output:
- Returns JSON dengan key `'prediction'`
- Nilai prediksi dalam format list (untuk mendukung batch prediction)

### Validasi Input:
- Data harus dalam format JSON yang valid
- Key `'data'` harus berisi array 2D dengan fitur-fitur numerik

---

## 🎓 Tentang Learning Path

Proyek ini adalah bagian dari **Dicoding - PIJAK by IBM** untuk pembelajaran:
- **Data Exploration & Analysis**
- **Feature Engineering**
- **Machine Learning Model Creation**
- **Model Deployment dengan REST API**
- **Integration dengan Flask Framework**

---

## 📚 Referensi & Resources

- [Kaggle - House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Scikit-learn Gradient Boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)
- [Joblib Serialization](https://joblib.readthedocs.io/)

---

## 📄 License

Project ini dilisensikan dibawah License yang ada di file `LICENSE`.

---

## 👤 annezetya

Dibuat sebagai bagian dari pembelajaran Machine Learning di Dicoding Academy.

**Happy Predicting! 🎉**

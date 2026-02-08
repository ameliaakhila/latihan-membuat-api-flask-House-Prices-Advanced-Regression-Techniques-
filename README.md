# House Prices Prediction API

Prediksi harga rumah menggunakan teknologi Machine Learning dan Flask REST API.

---

## 📌 Deskripsi Proyek

Proyek ini adalah implementasi **Machine Learning untuk prediksi harga rumah** berdasarkan fitur-fitur properti. Model yang dilatih kemudian di-deploy dalam bentuk **REST API menggunakan Flask** sehingga dapat diakses secara remote untuk melakukan prediksi harga rumah secara real-time.

### Tujuan Proyek:
1. **Training Model Machine Learning** - Melatih model Gradient Boosting Regressor pada dataset House Prices
2. **Feature Engineering** - Memproses dan mengoptimalkan fitur-fitur untuk meningkatkan akurasi prediksi
3. **API Development** - Membuat REST API menggunakan Flask untuk serve model prediction
4. **Production Ready** - Menyediakan endpoint yang dapat diintegrasikan dengan aplikasi lain

---

## 🎯 Objektif

API ini memungkinkan Anda untuk:
- ✅ Memprediksi harga rumah berdasarkan karakteristik properti
- ✅ Mengirim data dalam format JSON dan mendapatkan prediksi dengan cepat
- ✅ Mengintegrasikan model ML ke dalam aplikasi web atau sistem lain
- ✅ Melakukan prediksi multiple data dalam satu request

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

## 🚀 Cara Menggunakan API

### Prerequisites:
```bash
pip install -r requirement.txt
```

### Menjalankan Server:
```bash
python learn_api_flask.py
```

Server akan berjalan di: `http://localhost:5000`

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

### Menggunakan cURL:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [[1500, 3, 2, 2000, 8500, 1, 5, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
  }'
```

### Menggunakan Python (requests):
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

### Menggunakan JavaScript (Fetch API):
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

## 📁 Struktur File

```
.
├── Dicoding_Machine_Learning.ipynb    # Notebook untuk training dan EDA
├── learn_api_flask.py                  # Flask API application
├── gbr_model.joblib                    # Model machine learning (trained)
├── feature_columns.joblib              # Feature columns configuration
├── data.json                           # Sample data untuk testing
├── requirement.txt                     # Python dependencies
├── README.md                           # Dokumentasi ini
├── LICENSE                            # License file
└── dataset/
    ├── train.csv                       # Training dataset
    ├── test.csv                        # Test dataset
    ├── sample_submission.csv           # Sample submission format
    └── data_description.txt            # Deskripsi fitur-fitur
```

---

## 📦 Dependencies

Lihat `requirement.txt` untuk semua dependencies yang diperlukan:

- **Flask** - Web framework untuk membuat API
- **pandas** - Data manipulation dan analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning library
- **joblib** - Object serialization
- **scipy** - Scientific computing
- **seaborn & matplotlib** - Data visualization (development)

---

## 🔧 Instalasi & Setup Lengkap

### Step 1: Install Python

#### Windows:
1. Buka browser dan kunjungi [python.org](https://www.python.org/downloads/)
2. Download **Python 3.9 atau lebih baru** (Windows Installer)
3. Jalankan installer dan **PASTIKAN centang "Add Python to PATH"**
4. Klik "Install Now"

#### macOS:
```bash
# Menggunakan Homebrew
brew install python3
```

#### Linux (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv
```

### Step 2: Verifikasi Instalasi Python & Pip

Buka Command Prompt/Terminal dan ketik:

```bash
# Cek versi Python
python --version

# Cek versi pip
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

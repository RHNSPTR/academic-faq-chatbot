# 🎓 AcademiBot — Chatbot FAQ Akademik & Reservasi Konsultasi Dosen

> **Tugas UAS Mata Kuliah Natural Language Processing (NLP)**  
> Universitas Mercu Buana

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0+-000000?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-blue)

---

## 📖 Deskripsi

**AcademiBot** adalah chatbot berbasis web yang dirancang untuk membantu mahasiswa mendapatkan informasi seputar akademik kampus secara cepat dan interaktif. Chatbot ini dibangun menggunakan teknik **Natural Language Processing (NLP)** dan **Machine Learning** untuk memahami pertanyaan pengguna serta memberikan respons yang relevan.

### Fitur Utama

| Fitur | Deskripsi |
|---|---|
| 📋 **FAQ Pengisian KRS** | Informasi jadwal, portal SIAKAD, batas SKS berdasarkan IPS, dan prosedur KRS susulan |
| 🎓 **FAQ Sidang Tugas Akhir** | Syarat pendaftaran sidang, berkas yang diperlukan, dan prosedur pendaftaran |
| 📅 **Reservasi Bimbingan Dosen** | Sistem reservasi multi-turn dengan slot filling (dosen, tanggal, jam) |
| 💬 **Sapaan & Percakapan** | Respon interaktif untuk sapaan dan penutup percakapan |

---

## 👥 Anggota Kelompok

| No | Nama | NIM |
|:---:|---|---|
| 1 | **Rehan Saputra** | 41523010065 |
| 2 | **Muhammad Nurul Arsy** | 41523010100 |
| 3 | **Putri Amelia** | 41523010031 |

---

## 🏗️ Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER (Browser)                          │
│                     index.html + style.css                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │  HTTP POST /chat
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Flask Backend (app.py)                      │
│                                                                  │
│  ┌──────────────┐  ┌─────────────────┐  ┌────────────────────┐  │
│  │    Text       │  │  Intent          │  │  Dialog Manager    │  │
│  │ Preprocessing │─▶│  Classification  │─▶│  & State Machine   │  │
│  │              │  │  (model.pkl)     │  │  (Slot Filling)    │  │
│  └──────────────┘  └─────────────────┘  └────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│               Training Pipeline (train_model.py)                 │
│                                                                  │
│   dataset.json ──▶ Preprocessing ──▶ TF-IDF ──▶ Logistic Reg.  │
│                                                                  │
│   Output: model.pkl + vectorizer.pkl                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Pipeline NLP

### 1. Text Preprocessing
- **Case Folding** — Konversi teks ke huruf kecil (`lowercase`)
- **Cleaning** — Menghapus tanda baca, angka, dan karakter khusus
- **Tokenisasi** — Memecah teks menjadi token berdasarkan spasi

### 2. Feature Extraction
- **TF-IDF Vectorizer** dengan parameter:
  - `max_features=5000`
  - `ngram_range=(1, 2)` — Unigram & Bigram
  - `sublinear_tf=True`

### 3. Klasifikasi Intent
- **Algoritma:** Logistic Regression (`solver='lbfgs'`, `C=10`, `max_iter=1000`)
- **Intent yang didukung:**

| Intent | Deskripsi |
|---|---|
| `sapaan` | Salam pembuka dan penutup |
| `faq_krs` | Pertanyaan seputar KRS |
| `faq_sidang` | Pertanyaan seputar sidang tugas akhir |
| `reservasi_bimbingan` | Permintaan reservasi bimbingan dosen |

### 4. Dialog Manager & Slot Filling
Untuk intent `reservasi_bimbingan`, sistem menggunakan **Finite State Machine** dengan 5 state:

```
idle ──▶ filling_dosen ──▶ filling_tanggal ──▶ filling_jam ──▶ confirming
                                                                    │
                                                         Ya ──▶ ✅ Tersimpan
                                                        Tidak ──▶ ❌ Dibatalkan
```

**Slot yang diisi:**
| Slot | Contoh Input | Metode Ekstraksi |
|---|---|---|
| `dosen` | "Pak Andi", "Bu Sari" | Regex: pola `Pak/Bu/Dr/Prof + Nama` |
| `tanggal` | "Senin", "15/07/2025", "besok" | Regex: nama hari, format tanggal, kata kunci |
| `jam` | "10:00", "jam 2 siang", "pukul 14.00" | Regex: format waktu, konversi periode hari |

### 5. Evaluasi Model
- **Metrik:** Accuracy, Precision, Recall, F1-Score
- **Visualisasi:** Classification Report & Confusion Matrix

---

## 📂 Struktur Project

```
UAS_NLP/
├── app.py                  # Flask backend (dialog manager, slot filling, state machine)
├── train_model.py          # Pipeline training NLP (preprocessing, TF-IDF, evaluasi)
├── dataset.json            # Dataset utterances per intent
├── model.pkl               # Model Logistic Regression (hasil training)
├── vectorizer.pkl          # TF-IDF Vectorizer (hasil training)
├── chat_logs.json          # Log percakapan chatbot
├── requirements.txt        # Daftar dependensi Python
├── README.md               # Dokumentasi project
├── templates/
│   └── index.html          # Halaman web chatbot (UI)
└── static/
    └── css/
        └── style.css       # Stylesheet (Dark Mode Premium)
```

---

## ⚙️ Teknologi yang Digunakan

| Kategori | Teknologi |
|---|---|
| **Bahasa** | Python 3.10+ |
| **Web Framework** | Flask |
| **Machine Learning** | scikit-learn (Logistic Regression) |
| **Feature Extraction** | TF-IDF Vectorizer |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **UI Theme** | Dark Mode Premium (Biru Tua + Aksen Teal/Cyan) |
| **Font** | Google Fonts — Inter |

---

## 🚀 Cara Menjalankan

### Prasyarat
- Python 3.10 atau lebih baru
- pip (Python package manager)

### 1. Clone Repository

```bash
git clone https://github.com/RHNSPTR/academic-faq-chatbot.git
cd academic-faq-chatbot
```

### 2. Buat Virtual Environment (Opsional tapi Direkomendasikan)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependensi

```bash
pip install -r requirements.txt
```

### 4. Training Model

```bash
python train_model.py
```

Output yang dihasilkan:
- `model.pkl` — Model Logistic Regression yang sudah dilatih
- `vectorizer.pkl` — TF-IDF Vectorizer yang sudah di-fit
- Laporan evaluasi (accuracy, classification report, confusion matrix)

### 5. Jalankan Aplikasi

```bash
python app.py
```

### 6. Akses Chatbot

Buka browser dan akses:
```
http://127.0.0.1:5000
```

---

## 💬 Contoh Penggunaan

### FAQ KRS
```
User  : Bagaimana cara mengisi KRS?
Bot   : 📋 Informasi Pengisian KRS: ...
```

### FAQ Sidang
```
User  : Apa syarat pendaftaran sidang tugas akhir?
Bot   : 🎓 Informasi Sidang Tugas Akhir: ...
```

### Reservasi Bimbingan Dosen (Multi-Turn)
```
User  : Saya ingin reservasi bimbingan dosen
Bot   : 👨‍🏫 Dengan dosen siapa Anda ingin menjadwalkan bimbingan?
User  : Pak Andi
Bot   : 📅 Hari atau tanggal berapa Anda ingin bimbingan dengan Pak Andi?
User  : Senin
Bot   : 🕐 Jam berapa Anda ingin bimbingan pada hari Senin?
User  : Jam 10 pagi
Bot   : 📋 Konfirmasi Reservasi Bimbingan: Pak Andi, Senin, 10:00
        Apakah Anda yakin? (Ya/Tidak)
User  : Ya
Bot   : ✅ Reservasi Berhasil Disimpan!
```

---

## 📊 Komponen NLP yang Diterapkan

| No | Komponen NLP | Implementasi |
|:---:|---|---|
| 1 | Text Preprocessing | Case folding, cleaning, tokenisasi |
| 2 | Feature Extraction | TF-IDF Vectorizer (unigram + bigram) |
| 3 | Text Classification | Logistic Regression (intent classification) |
| 4 | Information Extraction | Rule-based regex (slot filling: dosen, tanggal, jam) |
| 5 | Dialog Management | Finite State Machine (multi-turn conversation) |
| 6 | Evaluation | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |

---

## 📝 Lisensi

Project ini dibuat untuk keperluan akademik sebagai Tugas UAS mata kuliah **Natural Language Processing** di **Universitas Mercu Buana**.

---

<div align="center">

**AcademiBot** — *Asisten Virtual untuk FAQ Akademik & Reservasi Konsultasi Dosen* 🎓

Dibuat dengan ❤️ oleh Kelompok 12 NLP — Universitas Mercu Buana

</div>

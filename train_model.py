"""
=============================================================================
  train_model.py - Pipeline NLP: Preprocessing, TF-IDF, Klasifikasi & Evaluasi
  AcademiBot: Chatbot FAQ Akademik dan Reservasi Konsultasi Dosen
=============================================================================
"""

import json
import re
import string
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ============================================================================
# 1. LOAD DATASET
# ============================================================================

def load_dataset(filepath="dataset.json"):
    """Memuat dataset dari file JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    texts = []
    labels = []
    for intent_group in data["intents"]:
        intent_label = intent_group["intent"]
        for utterance in intent_group["utterances"]:
            texts.append(utterance)
            labels.append(intent_label)
    
    return texts, labels


# ============================================================================
# 2. TEXT PREPROCESSING (Custom Functions)
# ============================================================================

def lowercase_text(text):
    """Mengubah teks menjadi huruf kecil."""
    return text.lower()


def clean_text(text):
    """Menghapus tanda baca, angka, dan karakter khusus dari teks."""
    # Hapus tanda baca
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Hapus angka
    text = re.sub(r"\d+", "", text)
    # Hapus karakter khusus dan whitespace berlebihan
    text = re.sub(r"[^\w\s]", "", text)
    # Hapus whitespace berlebihan
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text):
    """Tokenisasi sederhana: split berdasarkan spasi."""
    tokens = text.split()
    return tokens


def preprocess_text(text):
    """Pipeline preprocessing lengkap: lowercase -> cleaning -> tokenisasi -> gabung kembali."""
    text = lowercase_text(text)
    text = clean_text(text)
    tokens = tokenize_text(text)
    return " ".join(tokens)


# ============================================================================
# 3. ANALISIS DISTRIBUSI DATA
# ============================================================================

def print_data_distribution(labels):
    """Mencetak distribusi data per intent."""
    from collections import Counter
    counter = Counter(labels)
    total = len(labels)
    
    print("=" * 65)
    print("  ANALISIS DISTRIBUSI DATA PER INTENT")
    print("=" * 65)
    print(f"  {'Intent':<25} {'Jumlah':>8} {'Persentase':>12}")
    print("-" * 65)
    for intent, count in sorted(counter.items()):
        pct = (count / total) * 100
        bar = "#" * int(pct / 2)
        print(f"  {intent:<25} {count:>8} {pct:>10.1f}%  {bar}")
    print("-" * 65)
    print(f"  {'TOTAL':<25} {total:>8} {'100.0%':>12}")
    print("=" * 65)
    print()


# ============================================================================
# 4. TAMPILKAN CONTOH PREPROCESSING
# ============================================================================

def show_preprocessing_examples(texts, n=3):
    """Menampilkan contoh data sebelum dan sesudah preprocessing."""
    print("=" * 65)
    print("  CONTOH DATA SEBELUM DAN SESUDAH PREPROCESSING")
    print("=" * 65)
    
    # Pilih 3 contoh acak dari indeks berbeda
    np.random.seed(42)
    indices = np.random.choice(len(texts), size=min(n, len(texts)), replace=False)
    
    for i, idx in enumerate(indices, 1):
        original = texts[idx]
        processed = preprocess_text(original)
        tokens = tokenize_text(clean_text(lowercase_text(original)))
        
        print(f"\n  Contoh {i}:")
        print(f"  {'Sebelum':<15}: \"{original}\"")
        print(f"  {'Sesudah':<15}: \"{processed}\"")
        print(f"  {'Token':<15}: {tokens}")
    
    print()
    print("=" * 65)
    print()


# ============================================================================
# 5. TRAINING DAN EVALUASI MODEL
# ============================================================================

def train_and_evaluate():
    """Fungsi utama: load data, preprocessing, training, evaluasi, simpan model."""
    
    print()
    print("*" * 65)
    print("  AcademiBot - Pipeline NLP Training & Evaluasi")
    print("  Chatbot FAQ Akademik dan Reservasi Konsultasi Dosen")
    print("*" * 65)
    print()
    
    # --- Load Dataset ---
    print("[1/6] Memuat dataset...")
    texts, labels = load_dataset("dataset.json")
    print(f"      Dataset berhasil dimuat: {len(texts)} utterances\n")
    
    # --- Analisis Distribusi ---
    print("[2/6] Menganalisis distribusi data...")
    print_data_distribution(labels)
    
    # --- Contoh Preprocessing ---
    print("[3/6] Menampilkan contoh preprocessing...")
    show_preprocessing_examples(texts)
    
    # --- Preprocessing Seluruh Data ---
    print("[4/6] Melakukan preprocessing seluruh data...")
    texts_processed = [preprocess_text(t) for t in texts]
    print(f"      Preprocessing selesai untuk {len(texts_processed)} data.\n")
    
    # --- TF-IDF Vectorization ---
    print("[5/6] Membuat representasi TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(texts_processed)
    print(f"      Dimensi fitur TF-IDF: {X.shape}")
    print(f"      Jumlah vocabulary: {len(vectorizer.vocabulary_)}\n")
    
    # --- Train-Test Split ---
    print("[6/6] Melatih dan mengevaluasi model...\n")
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"      Data Training : {X_train.shape[0]} sampel")
    print(f"      Data Testing  : {X_test.shape[0]} sampel\n")
    
    # --- Training Model (Logistic Regression) ---
    model = LogisticRegression(
        max_iter=1000,
        C=10,
        solver="lbfgs",
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # --- Prediksi ---
    y_pred = model.predict(X_test)
    
    # --- Akurasi ---
    acc = accuracy_score(y_test, y_pred)
    print("=" * 65)
    print("  HASIL EVALUASI MODEL")
    print("=" * 65)
    print(f"\n  Accuracy: {acc:.4f} ({acc * 100:.2f}%)\n")
    
    # --- Classification Report ---
    print("-" * 65)
    print("  Classification Report (Precision, Recall, F1-Score)")
    print("-" * 65)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)
    
    # --- Confusion Matrix ---
    print("-" * 65)
    print("  Confusion Matrix")
    print("-" * 65)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(set(labels)))
    label_names = sorted(set(labels))
    
    # Header
    header = f"{'':>22}" + "".join(f"{name:>18}" for name in label_names)
    print(header)
    print(" " * 22 + "-" * (18 * len(label_names)))
    
    # Rows
    for i, name in enumerate(label_names):
        row = f"  {name:>18} |" + "".join(f"{cm[i][j]:>18}" for j in range(len(label_names)))
        print(row)
    
    print()
    print("  Keterangan: Baris = Label Aktual, Kolom = Label Prediksi")
    print("=" * 65)
    print()
    
    # --- Simpan Model dan Vectorizer ---
    print("  Menyimpan model dan vectorizer...")
    
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("  [OK] Model disimpan ke: model.pkl")
    
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print("  [OK] Vectorizer disimpan ke: vectorizer.pkl")
    
    print()
    print("*" * 65)
    print("  Pipeline training selesai! Model siap digunakan.")
    print("*" * 65)
    print()
    
    return model, vectorizer


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    train_and_evaluate()

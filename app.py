"""
=============================================================================
  app.py - Flask Backend: Dialog Manager, Slot Filling & State Machine
  AcademiBot: Chatbot FAQ Akademik dan Reservasi Konsultasi Dosen
=============================================================================
"""

import json
import os
import re
import pickle
import string
from datetime import datetime
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ============================================================================
# LOAD MODEL DAN VECTORIZER
# ============================================================================

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
CHAT_LOG_PATH = "chat_logs.json"

model = None
vectorizer = None


def load_model():
    """Memuat model dan vectorizer dari file pickle."""
    global model, vectorizer
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    print("[OK] Model dan Vectorizer berhasil dimuat.")


# ============================================================================
# TEXT PREPROCESSING (sama dengan train_model.py)
# ============================================================================

def preprocess_text(text):
    """Pipeline preprocessing: lowercase -> cleaning -> join."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================================
# INTENT CLASSIFICATION
# ============================================================================

def classify_intent(user_input):
    """Memprediksi intent dari input pengguna menggunakan model terlatih."""
    processed = preprocess_text(user_input)
    features = vectorizer.transform([processed])
    intent = model.predict(features)[0]
    confidence = max(model.predict_proba(features)[0])
    return intent, confidence


# ============================================================================
# RESPONSE TEMPLATES
# ============================================================================

RESPONSES = {
    "sapaan": [
        "Halo! 👋 Selamat datang di AcademiBot. Saya siap membantu Anda seputar akademik kampus. Silakan tanyakan tentang KRS, sidang tugas akhir, atau jadwalkan bimbingan dosen!",
        "Hai! 😊 Ada yang bisa saya bantu? Saya bisa menjawab pertanyaan seputar KRS, sidang, atau membantu reservasi bimbingan dosen.",
        "Terima kasih sudah menghubungi AcademiBot! Sampai jumpa dan semoga sukses! 🎓"
    ],
    "faq_krs": [
        "📋 **Informasi Pengisian KRS:**\n\n"
        "• **Jadwal:** Pengisian KRS biasanya dibuka H-7 sebelum perkuliahan dimulai. Cek kalender akademik untuk tanggal pastinya.\n"
        "• **Portal:** Akses melalui portal akademik mahasiswa (SIAKAD).\n"
        "• **Batas SKS:** Maksimal SKS ditentukan berdasarkan IPS semester lalu:\n"
        "  - IPS ≥ 3.00 → maks 24 SKS\n"
        "  - IPS 2.50 - 2.99 → maks 21 SKS\n"
        "  - IPS < 2.50 → maks 18 SKS\n"
        "• **Persetujuan:** KRS harus disetujui oleh Dosen Wali.\n"
        "• **KRS Susulan:** Hubungi Bagian Akademik dengan surat keterangan.\n\n"
        "Ada pertanyaan lain seputar KRS? 😊"
    ],
    "faq_sidang": [
        "🎓 **Informasi Sidang Tugas Akhir:**\n\n"
        "• **Syarat Umum:**\n"
        "  - Telah lulus semua mata kuliah wajib (minimal 140 SKS).\n"
        "  - Sudah menyelesaikan Seminar Proposal.\n"
        "  - Mendapat persetujuan dari Dosen Pembimbing.\n\n"
        "• **Berkas yang Diperlukan:**\n"
        "  - Formulir pendaftaran sidang\n"
        "  - Transkrip nilai sementara\n"
        "  - Kartu bimbingan yang sudah ditandatangani\n"
        "  - Laporan tugas akhir (3 rangkap)\n"
        "  - Bukti bebas tanggungan perpustakaan\n"
        "  - Surat keterangan bebas administrasi\n\n"
        "• **Pendaftaran:** Melalui Bagian Akademik atau portal online.\n"
        "• **Dress Code:** Kemeja rapi dan jas almamater.\n\n"
        "Ada pertanyaan lain seputar sidang? 📝"
    ],
}

GREETING_KEYWORDS = ["terima kasih", "makasih", "thanks", "bye", "sampai jumpa",
                      "dadah", "goodbye", "selamat tinggal", "cukup", "sudah"]

# ============================================================================
# SLOT FILLING & STATE MACHINE (Reservasi Bimbingan)
# ============================================================================

# Penyimpanan state per sesi (sederhana, in-memory)
user_sessions = {}


def get_session(session_id):
    """Mendapatkan atau membuat sesi baru untuk pengguna."""
    if session_id not in user_sessions:
        user_sessions[session_id] = {
            "state": "idle",        # idle | filling_dosen | filling_tanggal | filling_jam | confirming
            "slots": {
                "dosen": None,
                "tanggal": None,
                "jam": None,
            }
        }
    return user_sessions[session_id]


def reset_session(session_id):
    """Reset sesi pengguna ke keadaan awal."""
    user_sessions[session_id] = {
        "state": "idle",
        "slots": {
            "dosen": None,
            "tanggal": None,
            "jam": None,
        }
    }


def extract_dosen(text):
    """Ekstrak nama dosen dari teks menggunakan rule-based regex."""
    text_lower = text.lower()

    # Pola: "Pak/Bu/Bapak/Ibu [Nama]"
    pattern1 = re.search(r"(?:pak|bu|bapak|ibu|dr|prof|dosen)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)", text, re.IGNORECASE)
    if pattern1:
        return pattern1.group(0).strip().title()

    # Pola: "dengan/sama [Nama]"
    pattern2 = re.search(r"(?:dengan|sama|ke)\s+(?:pak|bu|bapak|ibu|dr|prof)?\s*([A-Za-z]+(?:\s+[A-Za-z]+)?)", text, re.IGNORECASE)
    if pattern2:
        name = pattern2.group(0).replace("dengan ", "").replace("sama ", "").replace("ke ", "").strip()
        return name.title()

    return None


def extract_tanggal(text):
    """Ekstrak tanggal/hari dari teks menggunakan rule-based regex."""
    text_lower = text.lower()

    # Pola: nama hari
    hari_pattern = re.search(
        r"\b(senin|selasa|rabu|kamis|jumat|jum'at|sabtu|minggu)\b",
        text_lower
    )
    if hari_pattern:
        return hari_pattern.group(1).capitalize()

    # Pola: tanggal format DD/MM/YYYY atau DD-MM-YYYY
    tanggal_pattern = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", text)
    if tanggal_pattern:
        return tanggal_pattern.group(1)

    # Pola: "tanggal [angka] [bulan]"
    tanggal_bulan = re.search(
        r"tanggal\s+(\d{1,2})\s+(januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)",
        text_lower
    )
    if tanggal_bulan:
        return f"{tanggal_bulan.group(1)} {tanggal_bulan.group(2).capitalize()}"

    # Pola: "besok", "lusa"
    if "besok" in text_lower:
        return "Besok"
    if "lusa" in text_lower:
        return "Lusa"

    return None


def extract_jam(text):
    """Ekstrak jam dari teks menggunakan rule-based regex."""
    text_lower = text.lower()

    # Pola: "HH:MM" atau "HH.MM"
    jam_pattern = re.search(r"\b(\d{1,2}[:.]\d{2})\b", text)
    if jam_pattern:
        return jam_pattern.group(1).replace(".", ":")

    # Pola: "jam [angka]" dengan opsional "pagi/siang/sore"
    jam_kata = re.search(r"jam\s+(\d{1,2})(?:\s*(pagi|siang|sore|malam))?", text_lower)
    if jam_kata:
        jam = int(jam_kata.group(1))
        periode = jam_kata.group(2) if jam_kata.group(2) else ""
        if periode == "siang" and jam < 12:
            jam += 12
        elif periode == "sore" and jam < 12:
            jam += 12
        elif periode == "malam" and jam < 12:
            jam += 12
        return f"{jam:02d}:00"

    # Pola: "pukul [angka]"
    pukul_pattern = re.search(r"pukul\s+(\d{1,2})[:.:]?(\d{2})?", text_lower)
    if pukul_pattern:
        jam = pukul_pattern.group(1)
        menit = pukul_pattern.group(2) if pukul_pattern.group(2) else "00"
        return f"{int(jam):02d}:{menit}"

    return None


def handle_reservation(user_input, session_id):
    """
    State Machine untuk proses reservasi bimbingan dosen.
    Mengelola multi-turn conversation dengan slot filling.
    """
    session = get_session(session_id)
    state = session["state"]
    slots = session["slots"]

    # ------ STATE: CONFIRMING ------
    if state == "confirming":
        lower_input = user_input.lower().strip()
        if lower_input in ["ya", "iya", "yap", "yes", "ok", "oke", "benar", "betul", "setuju", "y"]:
            # Simpan reservasi
            dosen = slots["dosen"]
            tanggal = slots["tanggal"]
            jam = slots["jam"]
            reset_session(session_id)
            return (
                f"✅ **Reservasi Berhasil Disimpan!**\n\n"
                f"📌 **Detail Reservasi:**\n"
                f"• Dosen: {dosen}\n"
                f"• Hari/Tanggal: {tanggal}\n"
                f"• Jam: {jam}\n\n"
                f"Silakan datang tepat waktu. Semoga bimbingannya lancar! 🎓"
            )
        elif lower_input in ["tidak", "no", "nggak", "enggak", "batal", "cancel", "n", "gak", "nope"]:
            reset_session(session_id)
            return "❌ Reservasi dibatalkan. Jika ingin menjadwalkan ulang, silakan beritahu saya kapan saja! 😊"
        else:
            return "Mohon jawab dengan **Ya** atau **Tidak** untuk mengonfirmasi reservasi. 🙏"

    # ------ AWAL: Coba ekstrak semua slot dari input awal ------
    if state == "idle":
        session["state"] = "filling_dosen"
        state = "filling_dosen"

        # Coba ekstrak semua slot dari pesan pertama
        dosen = extract_dosen(user_input)
        tanggal = extract_tanggal(user_input)
        jam = extract_jam(user_input)

        if dosen:
            slots["dosen"] = dosen
        if tanggal:
            slots["tanggal"] = tanggal
        if jam:
            slots["jam"] = jam

    # ------ STATE: FILLING_DOSEN ------
    if state == "filling_dosen":
        if not slots["dosen"]:
            dosen = extract_dosen(user_input)
            if dosen:
                slots["dosen"] = dosen
            else:
                # Jika input terlihat seperti nama (bukan perintah panjang)
                words = user_input.strip().split()
                if len(words) <= 4 and not any(kw in user_input.lower() for kw in ["jam", "tanggal", "hari", "pukul"]):
                    slots["dosen"] = user_input.strip().title()
                else:
                    return "👨‍🏫 Dengan dosen siapa Anda ingin menjadwalkan bimbingan?\n\n_(Contoh: Pak Andi, Bu Sari, Dr. Ahmad)_"

        # Cek slot tanggal dari input yang sama
        if not slots["tanggal"]:
            tanggal = extract_tanggal(user_input)
            if tanggal:
                slots["tanggal"] = tanggal

        # Cek slot jam dari input yang sama
        if not slots["jam"]:
            jam = extract_jam(user_input)
            if jam:
                slots["jam"] = jam

        session["state"] = "filling_tanggal"
        state = "filling_tanggal"

    # ------ STATE: FILLING_TANGGAL ------
    if state == "filling_tanggal":
        if not slots["tanggal"]:
            tanggal = extract_tanggal(user_input)
            if tanggal:
                slots["tanggal"] = tanggal
            else:
                return f"📅 Hari atau tanggal berapa Anda ingin bimbingan dengan **{slots['dosen']}**?\n\n_(Contoh: Senin, Rabu, tanggal 15 Juli, atau besok)_"

        # Cek slot jam dari input yang sama
        if not slots["jam"]:
            jam = extract_jam(user_input)
            if jam:
                slots["jam"] = jam

        session["state"] = "filling_jam"
        state = "filling_jam"

    # ------ STATE: FILLING_JAM ------
    if state == "filling_jam":
        if not slots["jam"]:
            jam = extract_jam(user_input)
            if jam:
                slots["jam"] = jam
            else:
                return f"🕐 Jam berapa Anda ingin bimbingan pada hari **{slots['tanggal']}**?\n\n_(Contoh: 10:00, jam 2 siang, pukul 14.00)_"

        # Semua slot terisi → masuk tahap konfirmasi
        session["state"] = "confirming"
        return (
            f"📋 **Konfirmasi Reservasi Bimbingan:**\n\n"
            f"• 👨‍🏫 Dosen: **{slots['dosen']}**\n"
            f"• 📅 Hari/Tanggal: **{slots['tanggal']}**\n"
            f"• 🕐 Jam: **{slots['jam']}**\n\n"
            f"Apakah Anda yakin ingin menjadwalkan bimbingan dengan **{slots['dosen']}** "
            f"pada hari **{slots['tanggal']}** jam **{slots['jam']}**?\n\n"
            f"_(Ketik: **Ya** / **Tidak**)_"
        )

    return "👨‍🏫 Dengan dosen siapa Anda ingin menjadwalkan bimbingan?\n\n_(Contoh: Pak Andi, Bu Sari, Dr. Ahmad)_"


# ============================================================================
# GENERATE RESPONSE
# ============================================================================

def generate_response(user_input, session_id="default"):
    """Menghasilkan respons berdasarkan input pengguna."""
    session = get_session(session_id)

    # Jika sedang dalam proses reservasi (state bukan idle), lanjutkan slot filling
    if session["state"] != "idle":
        return handle_reservation(user_input, session_id)

    # Klasifikasi intent
    intent, confidence = classify_intent(user_input)

    # Threshold confidence
    if confidence < 0.3:
        return "🤔 Maaf, saya kurang memahami pertanyaan Anda. Silakan coba tanyakan seputar:\n\n• 📋 Pengisian KRS\n• 🎓 Sidang Tugas Akhir\n• 👨‍🏫 Reservasi Bimbingan Dosen\n\nAtau ketik **\"halo\"** untuk memulai percakapan!"

    # Generate response berdasarkan intent
    if intent == "sapaan":
        lower_input = user_input.lower()
        if any(kw in lower_input for kw in GREETING_KEYWORDS):
            return RESPONSES["sapaan"][2]
        return RESPONSES["sapaan"][0]

    elif intent == "faq_krs":
        return RESPONSES["faq_krs"][0]

    elif intent == "faq_sidang":
        return RESPONSES["faq_sidang"][0]

    elif intent == "reservasi_bimbingan":
        return handle_reservation(user_input, session_id)

    return "🤔 Maaf, saya belum bisa memahami pertanyaan tersebut. Coba tanyakan tentang KRS, sidang, atau reservasi bimbingan."


# ============================================================================
# CHAT LOG
# ============================================================================

def save_chat_log(user_message, bot_response, session_id="default"):
    """Menyimpan log percakapan ke file JSON dengan timestamp."""
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": session_id,
        "user": user_message,
        "bot": bot_response,
    }

    # Load existing logs
    logs = []
    if os.path.exists(CHAT_LOG_PATH):
        try:
            with open(CHAT_LOG_PATH, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logs = []

    logs.append(log_entry)

    with open(CHAT_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)


# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route("/")
def index():
    """Halaman utama chatbot."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """API endpoint untuk menerima pesan dan mengirim respons."""
    data = request.get_json()
    user_message = data.get("message", "").strip()
    session_id = data.get("session_id", "default")

    if not user_message:
        return jsonify({"response": "Silakan ketik pesan Anda. 😊"})

    # Generate response
    bot_response = generate_response(user_message, session_id)

    # Simpan log percakapan
    save_chat_log(user_message, bot_response, session_id)

    return jsonify({"response": bot_response})


@app.route("/reset", methods=["POST"])
def reset():
    """Reset sesi pengguna."""
    data = request.get_json()
    session_id = data.get("session_id", "default")
    reset_session(session_id)
    return jsonify({"status": "ok", "message": "Sesi berhasil direset."})


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Load model saat startup
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        load_model()
    else:
        print("[WARNING] Model belum ditemukan! Jalankan train_model.py terlebih dahulu.")
        print("          Perintah: python train_model.py")
        exit(1)

    print("\n[SERVER] AcademiBot berjalan di http://127.0.0.1:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)

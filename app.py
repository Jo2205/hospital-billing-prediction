# Import library yang dibutuhkan
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Biaya Rumah Sakit",
    page_icon="🏥",
    layout="centered"
)


# --- Fungsi untuk Memuat Model ---
# Menggunakan cache agar model tidak perlu di-load ulang setiap kali ada interaksi
@st.cache_resource
def load_model_and_preprocessor():
    """Memuat model dan preprocessor yang sudah dilatih."""
    model = joblib.load('model_prediksi_biaya.joblib')
    preprocessor = joblib.load('preprocessor.joblib')
    return model, preprocessor

# Panggil fungsi untuk memuat model dan preprocessor
try:
    model, preprocessor = load_model_and_preprocessor()
    model_loaded = True
except FileNotFoundError:
    st.error("File model ('model_prediksi_biaya.joblib') atau preprocessor ('preprocessor.joblib') tidak ditemukan. Pastikan file tersebut ada di folder yang sama dengan app.py.")
    model_loaded = False


# --- Judul dan Deskripsi Aplikasi ---
st.title("Prediksi Biaya Tagihan Rumah Sakit 🏥")
st.write(
    "Aplikasi ini menggunakan model *machine learning* untuk memprediksi estimasi biaya "
    "tagihan pasien berdasarkan data yang Anda masukkan. Silakan isi form di bawah ini."
)


# --- Form Input dari Pengguna ---
st.header("Masukkan Data Pasien")

# Membuat layout dengan 2 kolom untuk kerapian
col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        label="Usia Pasien",
        min_value=0,
        max_value=120,
        value=45,  # Nilai default
        help="Masukkan usia pasien dalam tahun."
    )
    length_of_stay = st.number_input(
        label="Perkiraan Lama Rawat (Hari)",
        min_value=1,
        max_value=60,
        value=7, # Nilai default
        help="Masukkan perkiraan jumlah hari pasien akan dirawat."
    )
    gender = st.selectbox(
        label="Jenis Kelamin",
        options=['Female', 'Male']
    )
    blood_type = st.selectbox(
        label="Golongan Darah",
        options=['A+', 'B+', 'AB-', 'O+', 'A-', 'O-', 'B-', 'AB+']
    )
    medical_condition = st.selectbox(
        label="Kondisi Medis Utama",
        options=['Diabetes', 'Hypertension', 'Asthma', 'Arthritis', 'Obesity', 'Cancer', 'Bronchitis']
    )

with col2:
    insurance_provider = st.selectbox(
        label="Penyedia Asuransi",
        options=['Aetna', 'Blue Cross', 'Cigna', 'UnitedHealthcare', 'Medicare', 'MetLife']
    )
    admission_type = st.selectbox(
        label="Tipe Pendaftaran",
        options=['Emergency', 'Elective', 'Urgent']
    )
    medication = st.selectbox(
        label="Obat yang Diberikan",
        options=['Aspirin', 'Ibuprofen', 'Penicillin', 'Paracetamol', 'Lipitor', 'Zytiga']
    )
    test_results = st.selectbox(
        label="Hasil Tes",
        options=['Normal', 'Abnormal', 'Inconclusive']
    )

# --- Tombol untuk Melakukan Prediksi ---
st.divider() # Garis pemisah
predict_button = st.button("Hitung Prediksi Biaya", type="primary", use_container_width=True, disabled=not model_loaded)

# Blok ini HANYA akan berjalan jika tombol ditekan DAN model berhasil dimuat
if predict_button and model_loaded:
    # 1. Kumpulkan data input ke dalam DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Blood Type': [blood_type],
        'Medical Condition': [medical_condition],
        'Insurance Provider': [insurance_provider],
        'Admission Type': [admission_type],
        'Medication': [medication],
        'Test Results': [test_results],
        'Length of Stay': [length_of_stay]
    })

    try:
        # 2. Proses data input menggunakan preprocessor yang sudah ada
        input_processed = preprocessor.transform(input_data)
        
        # 3. Lakukan prediksi dengan model
        prediction = model.predict(input_processed)
        
        # 4. Tampilkan hasil prediksi dengan format yang rapi
        st.success(f"**Estimasi Biaya Tagihan: ${prediction[0]:,.2f}**")
        st.info("Catatan: Prediksi ini adalah estimasi berdasarkan data historis dan tidak menjamin biaya akhir.")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

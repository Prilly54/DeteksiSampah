import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Deteksi Sampah AI",
    page_icon="♻️",
    layout="centered"
)

# ==========================================
# 2. LOAD MODEL (Update Path Otomatis)
# ==========================================
import os # Tambahkan ini jika belum ada di paling atas

@st.cache_resource
def load_model():
    # Kode ini akan mencari file model RELATIF terhadap lokasi app.py
    # Jadi mau ditaruh di folder mana saja, dia akan tetap ketemu!
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model_efficientnetv2_sampah.keras")
    
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Model tidak ditemukan di: {model_path}.\nError: {e}")
        return None
# ==========================================
# 3. FUNGSI PREDIKSI
# ==========================================
def predict_image(image_input):
    # 1. Resize gambar
    image = image_input.resize(IMG_SIZE)
    # 2. Convert ke Array & Preprocessing
    img_array = np.array(image)
    # 3. Tambah dimensi batch (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 4. Prediksi
    predictions = model.predict(img_array)
    confidence = np.max(predictions[0])
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    
    return predicted_class, confidence

# ==========================================
# 4. TAMPILAN WEB (UI)
# ==========================================
st.title("♻️ Klasifikasi Sampah Cerdas")
st.markdown("Aplikasi ini menggunakan **EfficientNetV2** untuk mendeteksi sampah Metal, Paper, atau Plastic.")

# Tab untuk memilih metode input
tab1, tab2 = st.tabs(["📸 Ambil Foto", "📂 Upload Gambar"])

input_image = None

# --- TAB 1: KAMERA ---
with tab1:
    cam_image = st.camera_input("Ambil foto sampah")
    if cam_image:
        input_image = Image.open(cam_image)

# --- TAB 2: UPLOAD ---
with tab2:
    upload_file = st.file_uploader("Upload file gambar", type=["jpg", "png", "jpeg"])
    if upload_file:
        input_image = Image.open(upload_file)

# ==========================================
# 5. EKSEKUSI PREDIKSI
# ==========================================
if input_image is not None:
    st.divider()
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(input_image, caption="Gambar Input", use_column_width=True)
    
    with col2:
        st.subheader("Hasil Analisis:")
        
        if model:
            with st.spinner('Sedang memprediksi...'):
                label, score = predict_image(input_image)
            
            # Tampilkan Hasil dengan Warna
            if score > 0.60:
                st.success(f"**Jenis:** {label}")
                st.info(f"**Akurasi:** {score*100:.1f}%")
            else:
                st.warning(f"**Jenis:** {label} (?)")
                st.error(f"**Akurasi:** {score*100:.1f}% (Kurang yakin)")
                st.caption("⚠️ Objek mungkin tidak dikenal atau gambar kurang jelas.")
            
            # Progress Bar

            st.progress(int(score * 100))

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS
# ==========================================
st.set_page_config(
    page_title="Deteksi Sampah AI",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk mempercantik tampilan
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp > header {
        background-color: transparent;
    }
    h1 {
        color: #2E8B57;
        text-align: center;
        font-family: 'Helvetica', sans-serif;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD MODEL & KONFIGURASI
# ==========================================
CLASS_NAMES = ['Metal', 'Paper', 'Plastic']
IMG_SIZE = (224, 224)

# Database info tambahan untuk setiap jenis sampah
WASTE_INFO = {
    "Metal": {
        "icon": "🔧",
        "desc": "Logam/Kaleng",
        "tips": "Bersihkan sisa makanan/minuman. Kaleng bisa diremukkan untuk hemat tempat.",
        "color": "#e0e0e0" # Abu-abu
    },
    "Paper": {
        "icon": "📄",
        "desc": "Kertas/Karton",
        "tips": "Pastikan kertas kering dan tidak berminyak. Kardus sebaiknya dilipat.",
        "color": "#fff3cd" # Kuning muda
    },
    "Plastic": {
        "icon": "🥤",
        "desc": "Plastik",
        "tips": "Pisahkan tutup botol jika memungkinkan. Botol plastik bisa didaur ulang menjadi biji plastik.",
        "color": "#d1ecf1" # Biru muda
    }
}

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model_efficientnetv2_sampah.keras")
    
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Model tidak ditemukan di: {model_path}.\nError: {e}")
        return None

model = load_model()

# ==========================================
# 3. FUNGSI PREDIKSI
# ==========================================
def predict_image(image_input):
    image = image_input.resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    confidence = np.max(predictions[0])
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    
    return predicted_class, confidence

# ==========================================
# 4. SIDEBAR (MENU SAMPING)
# ==========================================
with st.sidebar:
    # --- LOGO PERMANEN ---
    # Menggunakan path dinamis agar gambar selalu ketemu
    base_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(base_dir, "logo.png") # Pastikan nama file di GitHub sama!
    
    # Cek apakah gambar ada (untuk menghindari error jika lupa upload)
    if os.path.exists(logo_path):
        st.image(logo_path, use_column_width=True)
    else:
        st.warning("Gambar logo.png belum diupload ke GitHub!")
    # ---------------------

    st.title("Tentang Aplikasi")
    st.info(
        """
        Aplikasi ini menggunakan Machine Learning (**EfficientNetV2**) 
        untuk mengklasifikasikan jenis sampah daur ulang.
        """
    )
    
    st.write("---")
    st.write("**Kategori yang dikenali:**")
    st.markdown("- 🔧 **Metal** (Logam)")
    st.markdown("- 📄 **Paper** (Kertas)")
    st.markdown("- 🥤 **Plastic** (Plastik)")
    
    st.write("---")
    confidence_threshold = st.slider("Ambang Batas Keyakinan (%)", 0, 100, 60, 5) / 100
    st.caption(f"sistem akan ragu jika akurasi di bawah {confidence_threshold*100:.0f}%")

# ==========================================
# 5. HALAMAN UTAMA (MAIN UI)
# ==========================================
st.title("♻️ Klasifikasi Sampah Cerdas")
st.markdown("### Upload foto sampahmu, biarkan sistem memilahnya!")
st.write("---")

# Tab Input
tab1, tab2 = st.tabs(["📸 Ambil Foto Langsung", "📂 Upload File Gambar"])

input_image = None

with tab1:
    cam_image = st.camera_input("Arahkan kamera ke objek sampah")
    if cam_image:
        input_image = Image.open(cam_image)

with tab2:
    upload_file = st.file_uploader("Seret & lepas gambar di sini", type=["jpg", "png", "jpeg"])
    if upload_file:
        input_image = Image.open(upload_file)

# ==========================================
# 6. HASIL & ANALISIS
# ==========================================
if input_image is not None:
    st.write("---")
    col1, col2 = st.columns([1, 1.5], gap="medium")
    
    with col1:
        st.image(input_image, caption="Gambar yang dianalisis", use_column_width=True, channels="RGB")
    
    with col2:
        if model:
            with st.spinner('🤖 Sedang menganalisis tekstur dan bentuk...'):
                label, score = predict_image(input_image)
            
            # Logika Tampilan Hasil
            info = WASTE_INFO.get(label, {})
            
            if score > confidence_threshold:
                # Tampilan Kotak Hasil Sukses
                st.markdown(
                    f"""
                    <div style="background-color: {info['color']}; padding: 20px; border-radius: 10px; border-left: 5px solid #2E8B57;">
                        <h2 style="color: #333; margin:0;">{info['icon']} {label}</h2>
                        <p style="margin:5px 0 0 0;">Kategori: <b>{info['desc']}</b></p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Metrik Akurasi
                st.write("") # Spacer
                st.metric(label="Tingkat Keyakinan", value=f"{score*100:.1f}%", delta="Sangat Yakin")
                
                # Tips Daur Ulang
                with st.expander("💡 Tips Pengelolaan", expanded=True):
                    st.write(info['tips'])
                    
            else:
                # Tampilan Jika AI Ragu
                st.error(f"🤔 Hmm, sistem kurang yakin.")
                st.metric(label="Prediksi Terdekat", value=f"{label}?", delta=f"{score*100:.1f}% (Rendah)", delta_color="inverse")
                st.warning("Objek tidak jelas, terlalu gelap, atau bukan jenis sampah yang dikenali.")

            # Progress bar visual
            st.write("Visualisasi Probabilitas:")
            st.progress(int(score * 100))
            
        else:
            st.error("Model gagal dimuat. Silakan cek log aplikasi.")







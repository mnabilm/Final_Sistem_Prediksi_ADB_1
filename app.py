import streamlit as st
import pandas as pd
import joblib

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Sistem Prediksi ADB",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    /* Improve overall spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Better divider styling */
    hr {
        margin: 1.5rem 0;
        border: none;
        border-top: 1px solid #e0e0e0;
    }
    
    /* Improve metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
    }
    
    /* Better button styling */
    .stButton > button {
        font-weight: 500;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    /* Input field improvements */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    /* Radio button spacing */
    .stRadio > div {
        gap: 1rem;
    }
    
    /* Better container borders */
    [data-testid="stVerticalBlock"] > div:has(> div.element-container) {
        border-radius: 12px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    /* Better alert boxes */
    .stAlert {
        border-radius: 8px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Load Model & Fitur
# =========================
model = joblib.load("model/model_rf_adb.pkl")
fitur = joblib.load("model/fitur_model.pkl")

# =========================
# Session State
# =========================
if "log_prediksi" not in st.session_state:
    st.session_state.log_prediksi = []

if "admin_login" not in st.session_state:
    st.session_state.admin_login = False

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.markdown("### 🩺 Sistem Prediksi ADB")
    st.markdown("---")
    aktor = st.radio(
        "**Pilih Mode Akses:**",
        ["User", "Admin"],
        label_visibility="visible"
    )
    
    st.markdown("---")
    st.markdown("##### 📋 Informasi Sistem")
    st.caption("Sistem ini menggunakan Machine Learning untuk memprediksi Anemia Defisiensi Besi berdasarkan parameter darah.")
    
    st.markdown("---")
    st.markdown("##### ⚠️ Disclaimer")
    st.caption("Hasil prediksi ini merupakan **indikasi awal** berdasarkan data yang diinputkan, bukan diagnosis medis. Untuk kepastian kondisi kesehatan, disarankan melakukan pemeriksaan lebih lanjut dengan tenaga kesehatan.")

# =====================================================
# ====================== USER =========================
# =====================================================
if aktor == "User":

    # Header Section
    st.markdown("# 🩺 Sistem Prediksi Anemia Defisiensi Besi")
    st.markdown("### Prediksi berbasis Machine Learning")
    st.markdown("---")

    # Info Box
    with st.expander("ℹ️ Cara Menggunakan Sistem", expanded=False):
        st.markdown("""
        **Langkah-langkah:**
        1. Pilih jenis kelamin pasien
        2. Masukkan hasil pemeriksaan darah lengkap
        3. Klik tombol **Prediksi** untuk melihat hasil
        
        **Parameter yang diperlukan:**
        - **HGB** (Hemoglobin): Kadar hemoglobin dalam darah
        - **MCV** (Mean Corpuscular Volume): Volume rata-rata sel darah merah
        - **MCH** (Mean Corpuscular Hemoglobin): Hemoglobin rata-rata per sel
        - **MCHC** (Mean Corpuscular Hemoglobin Concentration): Konsentrasi hemoglobin
        """)

    st.markdown("")

    # Input Section
    with st.container(border=True):
        st.markdown("#### 📝 Input Data Pemeriksaan")
        st.markdown("")
        
        # Gender Selection
        st.markdown("**Jenis Kelamin Pasien**")
        gender = st.radio(
            "Pilih jenis kelamin:",
            options=[0, 1],
            format_func=lambda x: "👩 Perempuan" if x == 0 else "👨 Laki-laki",
            horizontal=True,
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("**Parameter Darah Lengkap**")
        st.markdown("")

        # Row 1: HGB & MCV
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            hgb = st.number_input(
                "HGB (g/dL)",
                min_value=0.0,
                max_value=25.0,
                step=0.1,
                help="Hemoglobin: 12-16 g/dL (wanita), 13-17 g/dL (pria)"
            )
        with col2:
            mcv = st.number_input(
                "MCV (fL)",
                min_value=0.0,
                max_value=150.0,
                step=0.1,
                help="Mean Corpuscular Volume: 80-100 fL"
            )

        # Row 2: MCH & MCHC
        col3, col4 = st.columns(2, gap="medium")
        with col3:
            mch = st.number_input(
                "MCH (pg)",
                min_value=0.0,
                max_value=50.0,
                step=0.1,
                help="Mean Corpuscular Hemoglobin: 27-31 pg"
            )
        with col4:
            mchc = st.number_input(
                "MCHC (g/dL)",
                min_value=0.0,
                max_value=50.0,
                step=0.1,
                help="Mean Corpuscular Hemoglobin Concentration: 32-36 g/dL"
            )

    st.markdown("")

    # Predict Button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_btn = st.button("🔍 Lakukan Prediksi", use_container_width=True, type="primary")

    if predict_btn:

        if hgb == 0 or mcv == 0 or mch == 0 or mchc == 0:
            st.warning("⚠️ Mohon lengkapi seluruh data pemeriksaan sebelum melakukan prediksi.")
        else:

            # Create input dataframe
            data_input = pd.DataFrame([{
                "GENDER": gender,
                "HGB": hgb,
                "MCV": mcv,
                "MCH": mch,
                "MCHC": mchc
            }])

            data_input = data_input[fitur]

            # Prediction
            prediksi = model.predict(data_input)[0]
            probabilitas = model.predict_proba(data_input)[0][1]

            st.markdown("---")
            st.markdown("### 📊 Hasil Prediksi")
            st.markdown("")

            # Result display
            if prediksi == 1:
                hasil = "ADB"
                st.error("🔴 **Status: Terindikasi Anemia Defisiensi Besi (ADB)**")
                st.markdown("""
                <div style='background-color: #ffe6e6; padding: 1rem; border-radius: 8px; border-left: 4px solid #ff4444;'>
                    <p style='margin: 0; color: #cc0000;'><strong>📌 Informasi:</strong></p>
                    <p style='margin: 0.5rem 0 0 0; color: #660000;'>
                    Berdasarkan parameter yang diinputkan, sistem mendeteksi adanya indikasi anemia defisiensi besi. 
                    Disarankan untuk berkonsultasi dengan tenaga kesehatan guna evaluasi lebih lanjut dan mendapatkan penanganan yang sesuai.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                hasil = "Tidak ADB"
                st.success("🟢 **Status: Tidak Terindikasi Anemia Defisiensi Besi**")
                st.markdown("""
                <div style='background-color: #e6f7e6; padding: 1rem; border-radius: 8px; border-left: 4px solid #44cc44;'>
                    <p style='margin: 0; color: #006600;'><strong>📌 Informasi:</strong></p>
                    <p style='margin: 0.5rem 0 0 0; color: #004400;'>
                    Berdasarkan parameter yang diinputkan, sistem tidak mendeteksi indikasi anemia defisiensi besi. 
                    Tetap disarankan untuk menjaga pola hidup sehat dan melakukan pemeriksaan kesehatan secara berkala.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")
            
            # Confidence metric
            confidence_col1, confidence_col2, confidence_col3 = st.columns([1, 2, 1])
            with confidence_col2:
                st.metric(
                    label="Tingkat Keyakinan Model",
                    value=f"{probabilitas * 100:.2f}%",
                    help="Persentase keyakinan model terhadap hasil prediksi"
                )

            # Save to log
            st.session_state.log_prediksi.append({
                "Gender": "Laki-laki" if gender == 0 else "Laki-laki",
                "HGB": hgb,
                "MCV": mcv,
                "MCH": mch,
                "MCHC": mchc,
                "Prediksi": hasil,
                "Probabilitas": f"{probabilitas * 100:.2f}%"
            })

# =====================================================
# ====================== ADMIN ========================
# =====================================================
elif aktor == "Admin":

    if not st.session_state.admin_login:

        # Login Page
        st.markdown("# 🔐 Login Admin")
        st.markdown("### Akses Dashboard Administrator")
        st.markdown("---")
        
        col_login1, col_login2, col_login3 = st.columns([1, 2, 1])
        
        with col_login2:
            with st.container(border=True):
                st.markdown("#### Masukkan Kredensial")
                st.markdown("")
                
                username = st.text_input("👤 Username", placeholder="Masukkan username")
                password = st.text_input("🔒 Password", type="password", placeholder="Masukkan password")
                
                st.markdown("")
                
                if st.button("🚀 Login", use_container_width=True, type="primary"):
                    if username == "admin" and password == "admin123":
                        st.session_state.admin_login = True
                        st.success("✓ Login berhasil! Mengalihkan ke dashboard...")
                        st.rerun()
                    else:
                        st.error("✗ Username atau password salah. Silakan coba lagi.")

    else:

        # Admin Dashboard
        st.markdown("# 📊 Dashboard Administrator")
        
        # Logout button
        col_header1, col_header2 = st.columns([4, 1])
        with col_header2:
            if st.button("🚪 Logout", type="secondary"):
                st.session_state.admin_login = False
                st.rerun()

        st.markdown("---")

        data_log = pd.DataFrame(st.session_state.log_prediksi)

        if not data_log.empty:

            # Metrics Section
            total = len(data_log)
            adb = (data_log["Prediksi"] == "ADB").sum()
            tidak_adb = (data_log["Prediksi"] == "Tidak ADB").sum()
            
            st.markdown("### 📈 Ringkasan Statistik")
            st.markdown("")

            col1, col2, col3 = st.columns(3, gap="large")
            
            with col1:
                st.metric(
                    label="Total Prediksi",
                    value=total,
                    help="Total seluruh prediksi yang dilakukan"
                )
            
            with col2:
                percentage_adb = (adb / total * 100) if total > 0 else 0
                st.metric(
                    label="Terindikasi ADB",
                    value=adb,
                    delta=f"{percentage_adb:.1f}%",
                    delta_color="inverse",
                    help="Jumlah prediksi positif ADB"
                )
            
            with col3:
                percentage_not_adb = (tidak_adb / total * 100) if total > 0 else 0
                st.metric(
                    label="Tidak ADB",
                    value=tidak_adb,
                    delta=f"{percentage_not_adb:.1f}%",
                    delta_color="normal",
                    help="Jumlah prediksi negatif ADB"
                )

            st.markdown("---")

            # Chart Section
            st.markdown("### 📊 Distribusi Hasil Prediksi")
            st.markdown("")
            
            chart_col1, chart_col2, chart_col3 = st.columns([1, 3, 1])
            with chart_col2:
                st.bar_chart(
                    data_log["Prediksi"].value_counts(),
                    height=300,
                    use_container_width=True
                )

            st.markdown("---")

            # Data Table Section
            st.markdown("### 📋 Riwayat Prediksi Lengkap")
            st.markdown("")
            
            # Add filter options
            filter_col1, filter_col2 = st.columns([1, 1])
            with filter_col1:
                filter_prediksi = st.multiselect(
                    "Filter berdasarkan hasil:",
                    options=data_log["Prediksi"].unique(),
                    default=data_log["Prediksi"].unique()
                )
            with filter_col2:
                filter_gender = st.multiselect(
                    "Filter berdasarkan gender:",
                    options=data_log["Gender"].unique(),
                    default=data_log["Gender"].unique()
                )
            
            # Apply filters
            filtered_data = data_log[
                (data_log["Prediksi"].isin(filter_prediksi)) &
                (data_log["Gender"].isin(filter_gender))
            ]
            
            st.dataframe(
                filtered_data,
                use_container_width=True,
                hide_index=True,
                height=400
            )
            
            st.markdown("")
            st.caption(f"Menampilkan {len(filtered_data)} dari {total} data")

        else:

            st.info("📭 Belum ada data prediksi yang tersedia. Silakan lakukan prediksi terlebih dahulu pada mode User.")

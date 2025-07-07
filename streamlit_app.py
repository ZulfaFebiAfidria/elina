import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Harga Daging Ayam Broiler - Jawa Timur",
    page_icon="🍗",
    layout="wide"
)

st.title("📊 Dashboard Prediksi Harga Daging Ayam Broiler - Jawa Timur")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Dataset", 
    "⚙️ Preprocessing", 
    "📈 Visualisasi", 
    "🤖 Model", 
    "📉 Hasil Prediksi"
])

# ============================
# Tab 1 - Upload Dataset
# ============================
with tab1:
    st.header("📂 Dataset")

    required_columns = [
        'Date',
        'Harga Pakan Ternak Broiler',
        'Harga DOC Broiler',
        'Harga Jagung TK Peternak',
        'Harga Daging Ayam Broiler'
    ]

    uploaded_file = st.file_uploader("Upload Dataset Excel (.xlsx)", type=["xlsx"])

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)

            # Konversi Date
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

            # Konversi semua kolom harga ke numerik
            harga_cols = required_columns[1:]
            for col in harga_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Kolom berikut tidak ditemukan: {', '.join(missing_cols)}")
            else:
                st.session_state['df'] = df
                st.success("✅ Dataset berhasil diunggah!")

                st.write("### 🔍 Data Preview:")
                st.dataframe(df.head())

                with st.expander("📊 Deskripsi Statistik"):
                    st.write("#### 📈 Statistik Numerik")
                    st.dataframe(df[harga_cols].describe())

                    st.write("#### 📅 Statistik Kolom Tanggal")
                    st.dataframe(df[['Date']].describe(datetime_is_numeric=True))

        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")
    else:
        st.info("Silakan unggah file Excel yang sesuai format.")

# ============================
# Tab 2 - Preprocessing
# ============================
with tab2:
    st.header("⚙️ Preprocessing Data")

    if 'df' in st.session_state:
        df = st.session_state['df'].copy()

        st.subheader("1️⃣ Normalisasi Nama Kolom")
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        st.write("Kolom setelah dinormalisasi:")
        st.write(df.columns.tolist())

        df.rename(columns={
            'harga_pakan_ternak_broiler': 'pakan',
            'harga_doc_broiler': 'doc',
            'harga_jagung_tk_peternak': 'jagung',
            'harga_daging_ayam_broiler': 'daging',
            'date': 'tanggal'
        }, inplace=True)

        kolom_target = ['pakan', 'doc', 'jagung', 'daging']
        df[kolom_target] = df[kolom_target].apply(pd.to_numeric, errors='coerce')

        st.subheader("2️⃣ Penanganan Missing Values (Interpolasi + Fill)")
        df[kolom_target] = df[kolom_target].interpolate(method='linear')
        df[kolom_target] = df[kolom_target].fillna(method='ffill').fillna(method='bfill')
        st.dataframe(df[kolom_target].isna().sum())

        st.subheader("3️⃣ Deteksi Outlier dengan IQR")
        numerik_cols = df[kolom_target].select_dtypes(include=np.number).columns.tolist()
        Q1 = df[numerik_cols].quantile(0.25)
        Q3 = df[numerik_cols].quantile(0.75)
        IQR = Q3 - Q1

        outliers = (df[numerik_cols] < (Q1 - 1.5 * IQR)) | (df[numerik_cols] > (Q3 + 1.5 * IQR))
        st.write("Jumlah outlier per kolom:")
        st.dataframe(outliers.sum())

        fig_outlier, ax_outlier = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df[numerik_cols], orient='h', palette='Set2', ax=ax_outlier)
        ax_outlier.set_title("Boxplot Deteksi Outlier (IQR)")
        st.pyplot(fig_outlier)

        st.subheader("4️⃣ Transformasi Data (Log)")
        for col in numerik_cols:
            df[f"{col}_log"] = np.log(df[col])

        log_cols = [f"{col}_log" for col in numerik_cols]
        st.write("Preview Kolom Log:")
        st.dataframe(df[log_cols].head())

        fig_log, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs = axs.flatten()
        for i, col in enumerate(log_cols):
            sns.histplot(df[col], kde=True, color='skyblue', ax=axs[i])
            axs[i].set_title(f'Distribusi Log: {col}')
        plt.tight_layout()
        st.pyplot(fig_log)

        st.session_state['df_clean'] = df
    else:
        st.warning("Silakan unggah dataset terlebih dahulu di tab 📂 Dataset.")

# ============================
# Tab 3 - Visualisasi
# ============================
with tab3:
    st.header("📈 Visualisasi Dataset")

    if 'df_clean' in st.session_state:
        df = st.session_state['df_clean']

        st.subheader("Distribusi Harga Daging Ayam Broiler")
        fig1, ax1 = plt.subplots()
        sns.histplot(df['daging'], kde=True, ax=ax1)
        ax1.set_title("Distribusi Harga Daging")
        st.pyplot(fig1)

        st.subheader("Korelasi antar Fitur")
        fig2, ax2 = plt.subplots()
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax2)
        ax2.set_title("Heatmap Korelasi")
        st.pyplot(fig2)
    else:
        st.warning("Silakan lakukan preprocessing terlebih dahulu.")

# ============================
# Tab 4 - Model
# ============================
with tab4:
    st.header("🤖 Model")
    st.info("Model prediksi (XGBoost, Optuna, dll) akan ditambahkan di sini.")

# ============================
# Tab 5 - Prediksi (Simulasi)
# ============================
with tab5:
    st.header("📉 Hasil Prediksi")

    if 'df_clean' in st.session_state:
        df = st.session_state['df_clean']
        st.subheader("Prediksi Harga Daging Ayam (Simulasi)")

        df_pred = df.copy()
        df_pred['pred_xgb'] = df['daging'] * 0.95
        df_pred['pred_xgb_optuna'] = df['daging'] * 0.97

        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(df['tanggal'], df['daging'], label='Aktual', linewidth=2)
        ax3.plot(df['tanggal'], df_pred['pred_xgb'], label='Prediksi XGBoost', linestyle='--')
        ax3.plot(df['tanggal'], df_pred['pred_xgb_optuna'], label='Prediksi Optuna', linestyle='--')
        ax3.set_xlabel("Tanggal")
        ax3.set_ylabel("Harga")
        ax3.legend()
        ax3.set_title("Perbandingan Harga Aktual vs Prediksi")
        st.pyplot(fig3)
    else:
        st.warning("Silakan lakukan preprocessing terlebih dahulu.")

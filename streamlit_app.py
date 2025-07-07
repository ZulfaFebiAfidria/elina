import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Page setup
st.set_page_config(
    page_title="Prediksi Harga Daging Ayam Broiler - Jawa Timur",
    page_icon="🍗",
    layout="wide"
)

st.title("📊 Dashboard Prediksi Harga Daging Ayam Broiler - Jawa Timur")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📂 Dataset", 
    "⚙️ Preprocessing", 
    "📈 Visualisasi", 
    "📉 Hasil Prediksi"
])

# Tab 1 - Dataset
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
            missing_cols = [col for col in required_columns if col not in df.columns]

            if missing_cols:
                st.error(f"❌ Kolom berikut tidak ditemukan: {', '.join(missing_cols)}")
            else:
                # Konversi kolom angka ke numerik
                for col in required_columns[1:]:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

                st.session_state['df'] = df
                st.success("✅ Dataset berhasil dibaca!")
                st.write("Data Preview:")
                st.dataframe(df.head())

                with st.expander("📊 Deskripsi Statistik"):
                    st.write("#### Statistik Kolom Numerik")
                    st.dataframe(df[required_columns[1:]].describe())

                    st.write("#### Statistik Kolom Tanggal")
                    st.dataframe(df[['Date']].describe(datetime_is_numeric=True))

        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")
    else:
        st.info("Silakan upload file Excel dengan format yang benar.")

# Tab 2 - Preprocessing
with tab2:
    st.header("⚙️ Preprocessing Data")

    if 'df' in st.session_state:
        df = st.session_state['df'].copy()

        # Normalisasi nama kolom
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        df.rename(columns={
            'harga_pakan_ternak_broiler': 'pakan',
            'harga_doc_broiler': 'doc',
            'harga_jagung_tk_peternak': 'jagung',
            'harga_daging_ayam_broiler': 'daging',
            'date': 'tanggal'
        }, inplace=True)

        kolom_target = ['pakan', 'doc', 'jagung', 'daging']

        st.subheader("1️⃣ Penanganan Missing Values")
        df[kolom_target] = df[kolom_target].interpolate(method='linear')
        for col in kolom_target:
            df[col].fillna(method='ffill', inplace=True)
            df[col].fillna(method='bfill', inplace=True)

        st.write("Jumlah missing value setelah penanganan:")
        st.dataframe(df[kolom_target].isna().sum())

        st.subheader("2️⃣ Deteksi Outlier (IQR Method)")
        Q1 = df[kolom_target].quantile(0.25)
        Q3 = df[kolom_target].quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df[kolom_target] < (Q1 - 1.5 * IQR)) | (df[kolom_target] > (Q3 + 1.5 * IQR))
        st.write("Jumlah outlier per kolom:")
        st.dataframe(outliers.sum())

        fig_outlier, ax_outlier = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df[kolom_target], orient='h', palette='Set2', ax=ax_outlier)
        ax_outlier.set_title("Boxplot Outlier (IQR)")
        st.pyplot(fig_outlier)

        st.subheader("3️⃣ Transformasi Log")
        for col in kolom_target:
            df[f"{col}_log"] = np.log(df[col])

        st.dataframe(df[[f"{col}_log" for col in kolom_target]].head())

        # Simpan ke session
        st.session_state['df_clean'] = df
    else:
        st.warning("Silakan upload data terlebih dahulu di tab 📂 Dataset.")

# Tab 3 - Visualisasi
with tab3:
    st.header("📈 Visualisasi Dataset")

    if 'df_clean' in st.session_state:
        df = st.session_state['df_clean']

        st.subheader("Distribusi Harga Daging")
        fig1, ax1 = plt.subplots()
        sns.histplot(df['daging'], kde=True, ax=ax1)
        st.pyplot(fig1)

        st.subheader("Korelasi Antar Fitur")
        fig2, ax2 = plt.subplots()
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)
    else:
        st.warning("Lakukan preprocessing data terlebih dahulu.")

# Tab 4 - Hasil Prediksi
with tab4:
    st.header("📉 Hasil Prediksi (Simulasi)")

    if 'df_clean' in st.session_state:
        df = st.session_state['df_clean'].copy()

        st.subheader("Simulasi Prediksi Harga")
        df['prediksi_xgb'] = df['daging'] * 0.95
        df['prediksi_optuna'] = df['daging'] * 0.97

        fig3, ax3 = plt.subplots(figsize=(12, 5))
        ax3.plot(df['tanggal'], df['daging'], label='Aktual', linewidth=2)
        ax3.plot(df['tanggal'], df['prediksi_xgb'], label='XGBoost', linestyle='--')
        ax3.plot(df['tanggal'], df['prediksi_optuna'], label='Optuna+XGB', linestyle='--')
        ax3.set_title("Prediksi vs Aktual")
        ax3.set_xlabel("Tanggal")
        ax3.set_ylabel("Harga")
        ax3.legend()
        st.pyplot(fig3)
    else:
        st.warning("Data belum tersedia. Upload dan preprocessing dulu.")

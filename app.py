# --- Standard Library Imports ---
import json
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
#from langdetect import detect, LangDetectException
from PIL import Image
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from huggingface_hub import hf_hub_download

import streamlit as st
import pandas as pd
import numpy as np
# Hapus tensorflow dan ganti dengan torch
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import os

def preprocess_text(text, normalization_dict, stopwords_list):
    """
    Membersihkan dan menstandarkan teks input dari pengguna.
    """
    try:
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        words = text.split()
        normalized_words = [normalization_dict.get(word, word) for word in words]
        text = " ".join(normalized_words)
        words = text.split()
        filtered_words = [word for word in words if word not in stopwords_list]
        text = " ".join(filtered_words)
        return text
    except Exception as e:
        st.error(f"Error saat preprocessing: {e}")
        return ""

@st.cache_resource
def load_all_resources():
    """
    Memuat model, tokenizer, dan semua kamus yang dibutuhkan.
    """
    with st.spinner("Memuat model dan semua komponen... Mohon tunggu."):
        model_name = "nurulfauzh/nurul-fauziah-indobert"
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.eval()
        except Exception as e:
            st.error(f"Gagal memuat model '{model_name}'. Error: {e}")
            return None, None, {}, {}, {}
        
        try:
            df_norm = pd.read_csv('slang_indo.csv')
            normalization_dict = dict(zip(df_norm.iloc[:, 0], df_norm.iloc[:, 1]))
        except FileNotFoundError:
            st.warning("File 'slang_indo.csv' tidak ditemukan.")
            normalization_dict = {}

        try:
            with open('stopwordbahasa.csv', 'r', encoding='utf-8') as f:
                stopwords_list = set(f.read().splitlines())
        except FileNotFoundError:
            st.warning("File 'stopwordbahasa.csv' tidak ditemukan.")
            stopwords_list = set()
            
        try:
            df_lexicon = pd.read_csv('kamus_lexicon_.csv')
            lexicon = dict(zip(df_lexicon.word, df_lexicon.language))
        except FileNotFoundError:
            st.warning("File 'kamus_lexicon_.csv' tidak ditemukan.")
            lexicon = {}
            
    return model, tokenizer, normalization_dict, stopwords_list, lexicon

model, tokenizer, normalization_dict, stopwords_list, lexicon = load_all_resources()

@st.cache_data
def load_lexicon(file_path):
    """Memuat kamus dari file CSV dan mengubahnya menjadi dictionary."""
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        lexicon = dict(zip(df.word, df.language))
        return lexicon
    else:
        st.error(f"File kamus tidak ditemukan di '{file_path}'. Halaman 'Prediksi Sentimen' tidak akan bekerja.")
        return {}
    

def konversi_df_ke_csv_bytes(dataframe):
    """Mengonversi DataFrame Pandas ke bytes CSV untuk diunduh."""
    return dataframe.to_csv(index=False).encode('utf-8')


#logo
logo_path = "images/logos.png"
st.sidebar.image(logo_path, use_container_width=50)

sidebar = """
<style>
    /* 1. Mengatur latar utama menjadi putih dan teks default menjadi gelap */
    [data-testid="stAppViewContainer"] > .main, 
    .main .block-container {
        background-color: #82adb0 !important; /* Latar belakang putih untuk area utama */
    }
    .main .block-container {
        color: #262730 !important; 
    }
    /* Memastikan elemen teks umum di area utama juga berwarna gelap */
    .main .block-container p, 
    .main .block-container li,
    .main .block-container label,
    .main .block-container h1,
    .main .block-container h2,
    .main .block-container h3,
    .main .block-container h4,
    .main .block-container h5,
    .main .block-container h6 {
        color: #262730 !important;
    }

    /* latar sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(243, 248, 255) !important; 
    }

    /* teks */
    [data-testid="stSidebar"] * {
        color: #000000 !important; /* Teks Hitam */
    }
</style>
"""
st.markdown(sidebar, unsafe_allow_html=True)
try:
    with open("model_scores.json") as f:
        model_scores = json.load(f)
except FileNotFoundError:
    st.error("File 'model_scores.json' tidak ditemukan. Pastikan file tersebut ada.")
    model_scores = {} 

# sidebar
menu = st.sidebar.selectbox("Menu", ["Home", "Masukan Dataset", "Model", "Prediksi Sentimen"])


if menu == "Home":
    st.title("SentiLensAI: Analisis Sentimen Politik Indonesia - Prabowo Gibran")
    st.write(
        "Selamat Datang! Aplikasi ini dirancang untuk memfasilitasi proses analisis dan visualisasi sentimen terhadap opini publik mengenai pemerintahan Prabowo–Gibran. Sistem ini mendukung evaluasi berbagai model analisis sentimen dengan input berupa data teks dari media sosial dalam bahasa Indonesia, bahasa Inggris, serta campuran bahasa Indonesia dan Inggris yaitu code mixed text."
    )
    st.write("---")

    #gambar presiden dan wakil presiden
    foto_presiden = os.path.abspath("images/presiden_prabowo.jpeg")
    foto_wapres = os.path.abspath("images/wakil_presiden.jpeg")

    col1, col2 = st.columns(2)

    with col1:
        if os.path.exists(foto_presiden):
            st.image(foto_presiden, use_container_width=True)
        else:
            st.warning("Gambar Presiden tidak ditemukan di path: " + foto_presiden)

    with col2:
        if os.path.exists(foto_wapres):
            st.image(foto_wapres, use_container_width=True)
        else:
            st.warning("Gambar Wakil Presiden tidak ditemukan di path: " + foto_wapres)
    # --- BAGIAN BARU (PENGGANTI FITUR APLIKASI) ---
    st.header("Tentang Aplikasi")
    
    col_info1, col_info2 = st.columns(2, gap="large")

    with col_info1:
        with st.container(border=True):
            st.markdown("#### Apa itu SentiLensAI?")
            st.write(
                "Sebuah alat bantu berbasis web yang dirancang untuk menganalisis sentimen "
                "(positif atau negatif) dari teks opini publik. Aplikasi ini mampu memproses teks dalam "
                "Bahasa Indonesia, Bahasa Inggris, dan campuran keduanya (*Code-Mixed*)."
            )
            st.markdown("##### Fitur Unggulan:")
            st.markdown(
                """
                - **Visualisasi Data Interaktif**: Unggah dataset dan lihat distribusi sentimen serta *word cloud*.
                - **Perbandingan Kinerja Model**: Evaluasi berbagai model AI dengan metrik yang jelas.
                - **Prediksi Sentimen Real-Time**: Analisis sentimen dari satu kalimat secara langsung.
                """
            )

    with col_info2:
        with st.container(border=True):
            st.markdown("#### Teknologi yang Digunakan")
            st.markdown(
                """
                - **Model Utama**: `nurulfauzh/nurul-fauziah-indobert` (Model IndoBERT yang telah di-*fine-tuning*).
                - **Framework**: Aplikasi ini dibangun menggunakan **Streamlit**.
                - **Dataset Latih**: Model dilatih pada dataset opini dari media sosial X dalam bahasa Indonesia, Inggris, dan *Code-Mixed*.
                - **Library Inti**: Pemanrosesan data dan pemodelan ditenagai oleh **Hugging Face Transformers** dan **PyTorch**.
                """
            )

    st.info(
        "Aplikasi ini dikembangkan murni untuk tujuan penelitian dalam rangka penyelesaian tugas akhir dan "
        "tidak dimaksudkan untuk mendiskreditkan pihak manapun."
    )


    
# halaman Masukan Dataset
elif menu == "Masukan Dataset":
    st.header("Masukan Dataset")
    uploaded_file = st.file_uploader("Unggah dataset CSV Anda", type="csv")
    st.divider()

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Pratinjau Dataset")
            st.dataframe(df.head())

            # Tipe dataset berdasarkan nama file
            dataset_name = uploaded_file.name.lower()
            dataset_type = "unknown"
            if "code" in dataset_name or "mixed" in dataset_name:
                dataset_type = "codemixed"
            elif "indo" in dataset_name:
                dataset_type = "indo"
            elif "inggris" in dataset_name or "en" in dataset_name:
                dataset_type = "inggris"
            st.info(f"Tipe Dataset Terdeteksi (dari nama file): **{dataset_type.capitalize()}**")
            
            # Pembagian Dataset
            st.divider()
            st.subheader("Bagi dan Unduh Data Latih/Uji")
            if st.button("Bagi & Siapkan Data untuk Unduh"):
                kolom_target_untuk_split = 'sentiment'
                if kolom_target_untuk_split in df.columns and len(df) > 1:
                    try:
                        target_series = df[kolom_target_untuk_split]
                        can_stratify = target_series.nunique() > 1 and all(target_series.value_counts() > 1)
                        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=target_series if can_stratify else None)
                        st.success("Dataset berhasil dipisah!")
                        st.session_state['df_train_untuk_unduh'] = df_train
                        st.session_state['df_test_untuk_unduh'] = df_test
                    except Exception as e_split:
                        st.error(f"Gagal memisahkan data: {e_split}")
                else:
                    st.warning(f"Kolom '{kolom_target_untuk_split}' tidak ditemukan atau data tidak cukup.")
            
            col_unduh1, col_unduh2 = st.columns(2)
            with col_unduh1:
                if 'df_train_untuk_unduh' in st.session_state:
                    csv_train_bytes = konversi_df_ke_csv_bytes(st.session_state['df_train_untuk_unduh'])
                    st.download_button("Unduh Data Latih (.csv)", csv_train_bytes, "data_train.csv", "text/csv")
            with col_unduh2:
                if 'df_test_untuk_unduh' in st.session_state:
                    csv_test_bytes = konversi_df_ke_csv_bytes(st.session_state['df_test_untuk_unduh'])
                    st.download_button("Unduh Data Uji (.csv)", csv_test_bytes, "data_test.csv", "text/csv")
            
            # Visualisasi Distribusi Sentimen (Sepenuhnya Dinamis)
            st.divider()
            st.subheader("Visualisasi Distribusi Sentimen")
            
            if 'sentiment' in df.columns:
                sentiment_counts = df['sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['sentiment', 'jumlah']
                sentiment_map = {1: 'Positif', 0: 'Negatif', -1: 'Netral'}
                sentiment_counts['sentiment_label'] = sentiment_counts['sentiment'].map(sentiment_map).fillna(sentiment_counts['sentiment'])
                color_map = {'Positif': 'rgb(192, 201, 238)', 'Negatif': 'rgb(162, 170, 219)'}
                col_bar, col_pie = st.columns(2, gap="large")

                with col_bar:
                    st.markdown("##### Jumlah per Kategori")
                    fig_bar = px.bar(sentiment_counts, x='sentiment_label', y='jumlah', color='sentiment_label', color_discrete_map=color_map, text_auto=True)
                    fig_bar.update_layout(showlegend=False)
                    st.plotly_chart(fig_bar, use_container_width=True)

                with col_pie:
                    st.markdown("##### Proporsi per Kategori")
                    fig_pie = px.pie(sentiment_counts, names='sentiment_label', values='jumlah', hole=0.4, color='sentiment_label', color_discrete_map=color_map)
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(showlegend=False)
                    st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.warning("Kolom 'sentiment' tidak ditemukan untuk membuat grafik.")

            #wordcloud
            st.divider()
            st.subheader("Visualisasi WordCloud")
            
            kolom_teks = 'full_text'
            kolom_sentimen = 'sentiment'

            if kolom_teks in df.columns and kolom_sentimen in df.columns:
                # WordCloud Keseluruhan
                st.markdown("##### WordCloud Keseluruhan")
                all_text = " ".join(df[kolom_teks].astype(str).dropna().values)
                if all_text.strip():
                    wordcloud_all = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(all_text)
                    fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
                    ax_wc.imshow(wordcloud_all, interpolation='bilinear')
                    ax_wc.axis("off")
                    st.pyplot(fig_wc)
                else:
                    st.info("Tidak ada teks untuk membuat WordCloud keseluruhan.")

                # WordCloud Positif dan Negatif Berdampingan
                st.markdown("##### WordCloud per Sentimen")
                col_wc1, col_wc2 = st.columns(2, gap="large")

                with col_wc1:
                    st.markdown("**Positif**")
                    positive_text = " ".join(df[df[kolom_sentimen] == 1][kolom_teks].astype(str).dropna().values)
                    if positive_text.strip():
                        wordcloud_pos = WordCloud(width=800, height=600, background_color='white', colormap='Greens').generate(positive_text)
                        fig_pos, ax_pos = plt.subplots()
                        ax_pos.imshow(wordcloud_pos, interpolation='bilinear')
                        ax_pos.axis("off")
                        st.pyplot(fig_pos)
                    else:
                        st.info("Tidak ada teks sentimen positif.")
                
                with col_wc2:
                    st.markdown("**Negatif**")
                    negative_text = " ".join(df[df[kolom_sentimen] == 0][kolom_teks].astype(str).dropna().values)
                    if negative_text.strip():
                        wordcloud_neg = WordCloud(width=800, height=600, background_color='black', colormap='Reds').generate(negative_text)
                        fig_neg, ax_neg = plt.subplots()
                        ax_neg.imshow(wordcloud_neg, interpolation='bilinear')
                        ax_neg.axis("off")
                        st.pyplot(fig_neg)
                    else:
                        st.info("Tidak ada teks sentimen negatif.")
            else:
                st.warning(f"Kolom '{kolom_teks}' atau '{kolom_sentimen}' tidak ditemukan.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")
#===================================================================================================================================================#

# halaman Model
elif menu == "Model":
    st.title("Evaluasi dan Perbandingan Model")

    if not model_scores:
        st.error("Data skor model tidak dapat dimuat atau formatnya salah.")
    else:
        # --- Bagian Pemilihan ---
        st.subheader("Pilih Dataset dan Model")

        try:
            # --- LOGIKA BARU UNTUK MENGAMBIL SEMUA NAMA DATASET UNIK ---
            semua_dataset = set()
            for model_data in model_scores.values():
                for dataset_key in model_data.keys():
                    semua_dataset.add(dataset_key)
            
            # Urutkan hasilnya agar rapi
            dataset_options = sorted(list(semua_dataset))
            # -------------------------------------------------------------
        except Exception:
            dataset_options = []

        if not dataset_options:
            st.error("Format data di model_scores.json salah atau tidak ada data dataset.")
        else:
            pilihan_dataset = st.selectbox(
                "Pilih Dataset untuk dianalisis:",
                options=dataset_options
            )
            
            # Filter model berdasarkan dataset yang dipilih (logika ini sudah benar)
            model_tersedia = [
                model_name for model_name in model_scores 
                if pilihan_dataset in model_scores.get(model_name, {})
            ]

            if not model_tersedia:
                st.warning(f"Tidak ada model yang diuji pada dataset '{pilihan_dataset}'.")
            else:
                pilihan_models = st.multiselect(
                    f"Pilih satu atau lebih model yang diuji pada dataset '{pilihan_dataset}':",
                    options=sorted(model_tersedia)
                )

                st.divider()

                # --- Sisa kode untuk menampilkan detail dan grafik tetap sama persis ---
                if not pilihan_models:
                    st.warning("Silakan pilih minimal satu model untuk ditampilkan.")
                else:
                    for model_key in pilihan_models:
                        selected = model_scores.get(model_key, {}).get(pilihan_dataset)
                        
                        if selected:
                            st.header(f"Hasil untuk: {model_key} (Dataset: {pilihan_dataset})")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric(label="Akurasi", value=f"{selected.get('accuracy', 0):.2%}")
                            col2.metric(label="Precision", value=f"{selected.get('precision', 0):.2%}")
                            col3.metric(label="Recall", value=f"{selected.get('recall', 0):.2%}")
                            col4.metric(label="F1 Score", value=f"{selected.get('f1_score', 0):.2%}")

                            conf_matrix_path = f"images/confusion_matrix/{selected.get('conf_matrix', '')}"
                            if selected.get('conf_matrix') and os.path.exists(conf_matrix_path):
                                with st.expander("Lihat Confusion Matrix"):
                                    st.image(conf_matrix_path, use_container_width=True)
                            
                            st.divider()
                        else:
                            st.error(f"Data untuk model '{model_key}' pada dataset '{pilihan_dataset}' tidak ditemukan.")

                if len(pilihan_models) > 1 and pilihan_dataset:
                    st.header(f"Grafik Perbandingan Performa Model pada Dataset: {pilihan_dataset}")
                    # ... (sisa kode grafik tidak berubah) ...
                    comparison_data = []
                    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score']
                    for model_name in pilihan_models:
                        scores_for_dataset = model_scores.get(model_name, {}).get(pilihan_dataset)
                        if scores_for_dataset:
                            for metric in metrics_to_compare:
                                comparison_data.append({
                                    "Model": model_name, "Metrik": metric.capitalize(),
                                    "Skor": scores_for_dataset.get(metric, 0)
                                })
                    
                    if comparison_data:
                        df_comparison = pd.DataFrame(comparison_data)
                        fig = px.bar(df_comparison, x="Metrik", y="Skor", color="Model", barmode="group",
                                    title=f"Perbandingan Metrik Evaluasi pada Dataset {pilihan_dataset}",
                                    text_auto='.2%', labels={'Skor': 'Nilai Skor', 'Metrik': 'Metrik Evaluasi'})
                        fig.update_yaxes(range=[0, 1])
                        st.plotly_chart(fig, use_container_width=True)
#-------------------------------------------------------------------------------------------------------------------------------------------#

# halaman Prediksi Sentimen

elif menu == "Prediksi Sentimen":
    st.title("Prediksi Sentimen")
    

    user_input = st.text_area(
        "Masukkan teks untuk dianalisis sentimennya:", 
        height=150, 
        placeholder="Contoh: Kinerja pemerintah sangat bagus dan I love it!"
    )

    if st.button("Prediksi Sekarang", type="primary"):
        if not user_input.strip():
            st.warning("Mohon masukkan teks terlebih dahulu.")
        elif not model or not tokenizer:
            st.error("Model tidak berhasil dimuat. Aplikasi tidak dapat melanjutkan.")
        else:
            with st.spinner("Sedang menganalisis..."):
                try:
                    preprocessed_text = preprocess_text(user_input, normalization_dict, stopwords_list)
                    
                    inputs = tokenizer(preprocessed_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
                    predicted_class_id = np.argmax(probabilities)
                    confidence_score = probabilities[predicted_class_id]
                    
                    sentiment_labels = {0: "Negatif", 1: "Positif"}
                    sentiment_result = sentiment_labels.get(predicted_class_id, "Tidak Diketahui")

                    kata_pengguna = set(user_input.lower().split())
                    found_id = any(lexicon.get(kata) == 'id' for kata in kata_pengguna)
                    found_en = any(lexicon.get(kata) == 'en' for kata in kata_pengguna)
                    
                    if found_id and found_en:
                        lang_label = "Code-Mixed"
                    elif found_id:
                        lang_label = "Indonesia"
                    elif found_en:
                        lang_label = "Inggris"
                    else:
                        lang_label = "Code-Mixed"
                    
                    st.divider()
                    st.subheader("Hasil Analisis Model")
                    
                    with st.expander("Teks yang Diproses oleh Model"):
                        st.code(preprocessed_text or "(Teks habis setelah preprocessing)", language=None)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="Bahasa Terdeteksi", value=lang_label) 
                    with col2:
                        st.metric(label="Prediksi Sentimen", value=sentiment_result)
                    
                    st.progress(float(confidence_score), text=f"Tingkat Kepercayaan Model: {confidence_score:.1%}")

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat proses prediksi: {e}")

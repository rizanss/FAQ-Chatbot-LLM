import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import requests
from sentence_transformers import SentenceTransformer, util

# =======================================================================
#  SETUP & FUNGSI INTI
# =======================================================================

# @st.cache_resource akan menyimpan aset-aset ini di memori
# agar tidak di-load ulang setiap kali ada interaksi di web
@st.cache_resource
def load_assets():
    try:
        # Load dataset FAQ dari file CSV lokal
        df = pd.read_csv('Mental_Health_FAQ.csv')
        df.rename(columns={'Questions': 'pertanyaan', 'Answers': 'jawaban'}, inplace=True)
        # Hapus kolom ID jika ada
        if 'Question_ID' in df.columns:
            df = df.drop('Question_ID', axis=1)

        # Load model embedding
        embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        # Buat embeddings untuk semua pertanyaan di database
        # dan pastikan ada di CPU untuk kompatibilitas Streamlit
        question_embeddings = embedding_model.encode(df['pertanyaan'].tolist(), convert_to_tensor=True).cpu()
        
        st.success("Aset (model & data) berhasil di-load!")
        return df, embedding_model, question_embeddings
    except FileNotFoundError:
        st.error("Error: File 'Mental_Health_FAQ.csv' tidak ditemukan. Pastikan file ada di folder yang sama dengan app.py")
        return None, None, None
    except Exception as e:
        st.error(f"Error saat me-load aset: {e}")
        return None, None, None

def find_best_answer(user_question, df, embedding_model, question_embeddings):
    query_embedding = embedding_model.encode(user_question, convert_to_tensor=True).cpu()
    cosine_scores = util.cos_sim(query_embedding, question_embeddings)
    best_match_index = torch.argmax(cosine_scores).item()
    return df['jawaban'].iloc[best_match_index]

def generate_llm_response(user_question, context, api_key):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    prompt_template = """
    Anda adalah asisten AI 'Mentor AI' yang ramah dan empatik.
    Jawab pertanyaan pengguna tentang kesehatan mental HANYA berdasarkan KONTEKS yang diberikan.
    Jawab dengan gaya bahasa yang natural dan membantu dalam Bahasa Indonesia.

    KONTEKS: {context}
    PERTANYAAN: {question}

    JAWABAN ANDA:
    """
    prompt = prompt_template.format(context=context, question=user_question)
    
    payload = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 300
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"Maaf, terjadi error saat menghubungi AI. Coba beberapa saat lagi. ({e})"

# =======================================================================
#  LOGIKA TAMPILAN (INTERFACE) STREAMLIT
# =======================================================================

st.set_page_config(page_title="Mentor AI Chatbot", page_icon="🫂")

# --- SIDEBAR ---
with st.sidebar:
    st.title("🫂 Mentor AI Chatbot")
    st.markdown("Ini adalah chatbot yang dirancang untuk menjawab pertanyaan seputar kesehatan mental.")
    st.markdown("Chatbot ini menggunakan **Retrieval-Augmented Generation (RAG)**:")
    st.markdown("1. **Mencari** jawaban paling relevan dari database FAQ.")
    st.markdown("2. **Menggunakan LLM (Llama3 via Groq)** untuk membuat jawaban baru yang lebih personal berdasarkan konteks yang ditemukan.")
    st.divider()
    # Ambil Groq API Key dari Streamlit Secrets
    groq_api_key = st.text_input("Masukkan Groq API Key Anda:", type="password", help="Daftar gratis di console.groq.com")
    
# --- MAIN CHAT INTERFACE ---
st.header("Tanyakan Apapun Tentang Kesehatan Mental")

# Inisialisasi chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Terima input dari user
if prompt := st.chat_input("Bagaimana perasaanmu hari ini?"):
    # Pastikan API key sudah dimasukkan
    if not groq_api_key:
        st.warning("Mohon masukkan Groq API Key Anda di sidebar terlebih dahulu.")
        st.stop()

    # Tampilkan pertanyaan user di chat
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Tambahkan pertanyaan user ke history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Tampilkan spinner dan generate jawaban
    with st.spinner("Mentor AI sedang merangkai kata..."):
        # Load aset
        df, embedding_model, question_embeddings = load_assets()
        if df is not None:
            context = find_best_answer(prompt, df, embedding_model, question_embeddings)
            response = generate_llm_response(prompt, context, groq_api_key)
            
            # Tampilkan jawaban dari asisten
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # Tambahkan jawaban asisten ke history
            st.session_state.messages.append({"role": "assistant", "content": response})
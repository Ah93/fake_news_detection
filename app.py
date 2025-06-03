# app.py
import streamlit as st
import torch
import pickle
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from model import TransformerGRUClassifier
from utils import clean_text

# Constants
MAX_SEQ_LEN = 100  # Ensure this matches what was used in training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and embedding matrix
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

embedding_matrix = np.load("embedding_matrix.npy")

# Initialize and load model
model = TransformerGRUClassifier(embedding_matrix)
model.load_state_dict(torch.load("fake_news_model.pt", map_location=device))
model.to(device)
model.eval()

# Streamlit page config
st.set_page_config(page_title="📰 Fake News Detector", page_icon="🧠", layout="centered")

# Custom CSS
st.markdown("""
    <style>
        .main {
            background-color: #0e1117;
            color: #FAFAFA;
        }
        .stTextArea textarea {
            background-color: #1e222a;
            color: #ffffff;
        }
        .stButton>button {
            background-color: #6c63ff;
            color: white;
            border-radius: 10px;
            padding: 0.5em 1em;
        }
        .stMarkdown h3 {
            color: #6c63ff;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Fake News Detector")
st.markdown("Check if a news statement is **real** or **fake** using a neural network model trained on the LIAR and ISOT datasets.")

def predict(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_SEQ_LEN)
    input_tensor = torch.tensor(padded, dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred = torch.argmax(output, dim=1).item()
    return pred, prob

# User input
user_input = st.text_area("✍️ Enter a news statement below:", height=150)

if st.button("🔎 Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a news statement.")
    else:
        with st.spinner("Analyzing the news... 🧠"):
            label, prob = predict(user_input)

        label_text = "✅ REAL" if label == 1 else "❌ FAKE"
        confidence = prob[label]

        st.markdown(f"### 🧾 Prediction: {label_text}")
        st.markdown(f"Confidence: **{confidence:.2%}**")

        st.progress(int(confidence * 100))

        st.markdown("---")
        st.subheader("📊 Probability Distribution")
        st.write({
            "FAKE": f"{prob[0]:.2%}",
            "REAL": f"{prob[1]:.2%}"
        })

        st.markdown("---")

# About section
with st.expander("ℹ️ About This App"):
    st.write("""
        This app uses a deep learning model (Transformer + BiGRU) trained on the LIAR and ISOT fake news datasets.
        It preprocesses text using tokenization, padding, and word embeddings, then classifies the input into FAKE or REAL. The model overall accuracy is: 89%.
    """)
    st.write("Made By Ahmed Sheikh.")

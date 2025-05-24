import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import time

# Load models
model_lstm = load_model('E:/ML/model_lstm.keras')
model_cnn = load_model('E:/ML/model_cnn.keras')

# Load tokenizer
with open('E:/ML/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_len = 100

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\S+|#\S+|RT", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    return padded

# Page config
st.set_page_config(
    page_title="🔥 Hate Speech Detector 🔥",
    page_icon="🛡️",
    layout="centered",
)

# Custom CSS for styling
st.markdown("""
    <style>
    .title {
        font-size: 48px;
        font-weight: bold;
        color: #d90429;
        text-align: center;
        margin-bottom: 0.2em;
    }
    .subtitle {
        font-size: 20px;
        color: #720026;
        text-align: center;
        margin-top: 0;
        margin-bottom: 2em;
    }
    .result-box {
        background: linear-gradient(135deg, #ffafbd, #ffc3a0);
        border-radius: 15px;
        padding: 20px;
        margin-top: 1em;
        text-align: center;
        font-size: 22px;
        font-weight: 600;
        color: #4a148c;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .progress-label {
        font-weight: 600;
        font-size: 18px;
        margin-top: 10px;
        color: #2b2d42;
    }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown('<div class="title">🔥 Hate Speech Detector 🔥</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by LSTM & CNN deep learning models</div>', unsafe_allow_html=True)

# Input area
user_input = st.text_area("📝 Enter your tweet or comment here:", height=130, placeholder="Type or paste text here...")

if st.button("🔍 Analyze"):

    if not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze!")
    else:
        # Show spinner and progress bar
        with st.spinner("Processing... Please wait! ⏳"):
            progress_bar = st.progress(0)
            for i in range(1, 101, 10):
                time.sleep(0.05)
                progress_bar.progress(i)

            processed_input = preprocess(user_input)
            pred_lstm = model_lstm.predict(processed_input)[0][0]
            pred_cnn = model_cnn.predict(processed_input)[0][0]

            # Force detection if certain offensive words appear
            offensive_keywords = {"fuck", "bitch", "nigga", "nigger", "hoe", "slut"}
            text_lower = user_input.lower()
            if any(word in text_lower for word in offensive_keywords):
                pred_lstm = pred_cnn = 0.6  # Force "⚠️ Possibly Offensive"

        # Interpret predictions
        def interpret_score(score):
            if score > 0.7:
                return "🚨 Hate Speech Detected!", "#d90429"
            elif score > 0.4:
                return "⚠️ Possibly Offensive", "#f8961e"
            else:
                return "✅ Clean / Not Hate Speech", "#43aa8b"

        lstm_msg, lstm_color = interpret_score(pred_lstm)
        cnn_msg, cnn_color = interpret_score(pred_cnn)

        # Show results side by side
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f'<div class="result-box" style="color:{lstm_color};">💡 <b>LSTM Model:</b><br>{lstm_msg}<br>Score: {pred_lstm:.3f}</div>', unsafe_allow_html=True)

        with col2:
            st.markdown(f'<div class="result-box" style="color:{cnn_color};">💡 <b>CNN Model:</b><br>{cnn_msg}<br>Score: {pred_cnn:.3f}</div>', unsafe_allow_html=True)

        # Average score and verdict
        avg_score = (pred_lstm + pred_cnn) / 2
        verdict, verdict_color = interpret_score(avg_score)

        st.markdown(f'<div class="result-box" style="background: linear-gradient(135deg, #90ee90, #32cd32); color: black; font-size: 24px;">🎯 <b>Final Verdict:</b><br>{verdict}<br>Average Score: {avg_score:.3f}</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<footer style="text-align:center; margin-top:4em; color:gray;">
    Built with ❤️ using Streamlit & TensorFlow | Developed by Sri Harshavardhan
</footer>
""", unsafe_allow_html=True)

import streamlit as st
import pickle
import string
import nltk
import matplotlib.pyplot as plt
import pandas as pd
import pytesseract
import re
from PIL import Image
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------- CONFIG ----------
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
st.set_page_config(page_title="Message Shield", page_icon="🛡️", layout="wide")

# ---------- 🔥 UI STYLE (ADDED ONLY) ----------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #020617, #0f172a);
    color: white;
}

.title {
    font-size: 48px;
    text-align: center;
    font-weight: bold;
    background: linear-gradient(90deg,#22c55e,#4ade80);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: glow 2s infinite alternate;
}

@keyframes glow {
    from {text-shadow: 0 0 10px #22c55e;}
    to {text-shadow: 0 0 25px #4ade80;}
}

/* CARD */
.card {
    background: rgba(30,41,59,0.6);
    backdrop-filter: blur(10px);
    padding: 20px;
    border-radius: 18px;
    margin-bottom: 15px;
    transition: 0.3s;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0px 0px 25px rgba(34,197,94,0.5);
}

/* BUTTON */
div.stButton > button {
    background: linear-gradient(135deg,#22c55e,#4ade80);
    color: white;
    border-radius: 12px;
    padding: 10px 20px;
    font-weight: bold;
}

div.stButton > button:hover {
    transform: scale(1.08);
    box-shadow: 0px 0px 20px rgba(34,197,94,0.8);
}

/* DASHBOARD */
.metric-box {
    background: rgba(30,41,59,0.7);
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    transition: 0.3s;
}

.metric-box:hover {
    transform: scale(1.05);
    box-shadow: 0px 0px 25px rgba(34,197,94,0.6);
}

.metric-value {
    font-size: 30px;
    font-weight: bold;
    color: #22c55e;
}

</style>
""", unsafe_allow_html=True)

# ---------- LOAD ----------
nltk.download('punkt')
nltk.download('stopwords')

vectorizer = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

ps = PorterStemmer()

# ---------- FUNCTIONS ----------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    words = [i for i in text if i.isalnum()]
    stop_words = set(stopwords.words('english'))
    return " ".join([ps.stem(i) for i in words if i not in stop_words])

def adjust_prediction(text, prob):
    if "http" in text or "www" in text:
        prob += 0.15
    return min(prob, 1)

# ---------- SESSION ----------
if "history" not in st.session_state:
    st.session_state.history = []

if "total" not in st.session_state:
    st.session_state.total = 0
    st.session_state.spam = 0
    st.session_state.safe = 0

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ---------- HEADER ----------
st.markdown('<div class="title">🛡️ Message Shield</div>', unsafe_allow_html=True)
st.write("Smart AI Spam Detection System")

# ---------- DASHBOARD (UPDATED UI ONLY) ----------
st.subheader("📊 Dashboard")

c1, c2, c3 = st.columns(3)

c1.markdown(f"<div class='metric-box'>Total<br><div class='metric-value'>{st.session_state.total}</div></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='metric-box'>Spam<br><div class='metric-value'>{st.session_state.spam}</div></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='metric-box'>Safe<br><div class='metric-value'>{st.session_state.safe}</div></div>", unsafe_allow_html=True)

st.markdown("---")

left, right = st.columns([2, 1])

# ================= LEFT =================
with left:

    # TEXT
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("✍️ Text Detection")

    text_input = st.text_area("Enter Message")

    if st.button("Check Text"):
        if text_input.strip():
            vec = vectorizer.transform([transform_text(text_input)])
            prob = model.predict_proba(vec)[0]

            spam_prob = adjust_prediction(text_input, prob[1])
            result = "Spam" if spam_prob > 0.5 else "Safe"

            st.session_state.last_result = (spam_prob, 1 - spam_prob)

            st.session_state.total += 1
            if result == "Spam":
                st.session_state.spam += 1
                st.error("🚫 Spam Message")
            else:
                st.session_state.safe += 1
                st.success("✅ Safe Message")

            st.session_state.history.append(("Text", result))
    st.markdown('</div>', unsafe_allow_html=True)

    # IMAGE
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📷 Image Detection")

    img_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

    if st.button("Check Image"):
        if img_file:
            image = Image.open(img_file)
            text = pytesseract.image_to_string(image)

            vec = vectorizer.transform([transform_text(text)])
            prob = model.predict_proba(vec)[0]

            spam_prob = adjust_prediction(text, prob[1])
            result = "Spam" if spam_prob > 0.5 else "Safe"

            st.session_state.last_result = (spam_prob, 1 - spam_prob)

            st.session_state.total += 1
            if result == "Spam":
                st.session_state.spam += 1
                st.error("🚫 Spam Image")
            else:
                st.session_state.safe += 1
                st.success("✅ Safe Image")

            st.session_state.history.append(("Image", result))
    st.markdown('</div>', unsafe_allow_html=True)

    # CSV
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📂 CSV Detection")

    csv_file = st.file_uploader("Upload CSV", type=["csv"])

    if st.button("Check CSV"):
        if csv_file:
            df = pd.read_csv(csv_file, on_bad_lines='skip')

            if 'message' not in df.columns:
                df.columns = ['message']

            df['message'] = df['message'].astype(str)

            vec = vectorizer.transform(df['message'].apply(transform_text))
            probs = model.predict_proba(vec)

            df['prediction'] = ["Spam" if p[1] > 0.5 else "Safe" for p in probs]

            st.dataframe(df)

            csv = df.to_csv(index=False).encode()
            st.download_button("Download Results", csv, "results.csv")

            spam_ratio = (df['prediction'] == "Spam").sum() / len(df)
            result = "Spam" if spam_ratio > 0.5 else "Safe"

            st.session_state.last_result = (spam_ratio, 1 - spam_ratio)

            st.session_state.total += 1
            if result == "Spam":
                st.session_state.spam += 1
            else:
                st.session_state.safe += 1

            st.session_state.history.append(("CSV", result))
    st.markdown('</div>', unsafe_allow_html=True)

# ================= RIGHT =================
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Insights")

    if st.session_state.last_result:
        spam_prob, safe_prob = st.session_state.last_result

        fig, ax = plt.subplots()
        if spam_prob == 0: spam_prob = 0.0001
        if safe_prob == 0: safe_prob = 0.0001

        ax.pie([spam_prob, safe_prob], labels=["Spam", "Safe"], autopct="%0.1f%%")
        st.pyplot(fig)
    else:
        st.info("Run detection to see insights")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- HISTORY ----------
st.markdown("---")
st.subheader("📜 History")

if st.session_state.history:
    for item, res in st.session_state.history[::-1]:
        color = "#ef4444" if res == "Spam" else "#22c55e"
        st.markdown(f"""
        <div style="background:#1e293b;padding:10px;border-left:5px solid {color};margin-bottom:8px">
        {item} → <b style="color:{color}">{res}</b>
        </div>
.        """, unsafe_allow_html=True)
else:
    st.info("No history yet")

st.markdown("---")
st.write("🚀 Final Stable Spam Detection System")
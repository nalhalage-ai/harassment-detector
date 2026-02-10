import streamlit as st
from classifier import classify
from guidance import get_guidance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

texts = [...]
labels = [...]

vec = TfidfVectorizer(ngram_range=(1,2))
X = vec.fit_transform(texts)

model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

with open("model/model.pkl", "wb") as f:
    pickle.dump((model, vec), f)

st.set_page_config(page_title="Harassment Detector", layout="centered")

st.title("🛡️ Gender-Inclusive Harassment Detection")
st.caption("Describe what happened. All genders supported.")

text = st.text_area("Incident description", height=150)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please describe the incident.")
    else:
        label, confidence = classify(text)
        st.subheader(f"Result: {label}")
        st.progress(confidence)

        info = get_guidance(label)

        if "law" in info:
            st.error("⚠️ Serious Case Detected")
            st.write("**Relevant Laws:**", info["law"])
            st.write("**Evidence Tips:**")
            for tip in info["tips"]:
                st.write("•", tip)
            st.write("**SOS Contacts:**", ", ".join(info["sos"]))
        else:
            st.success(info["message"])

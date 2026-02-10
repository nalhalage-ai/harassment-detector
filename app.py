import streamlit as st
from classifier import classify
from guidance import get_guidance
import os

st.set_page_config(
    page_title="Harassment Detection System",
    layout="centered"
)

st.title("🛡️ Gender-Inclusive Harassment Detection")
st.caption("Describe the incident in your own words. All genders supported.")

# Inform about ML status
if not os.path.exists("model/model.pkl"):
    st.info(
        "⚠️ ML model not found. "
        "App is running in rule-based + hybrid fallback mode."
    )

text = st.text_area(
    "Incident description",
    height=160,
    placeholder="Example: He keeps messaging me every night even after I asked him to stop."
)

if st.button("Analyze Incident"):
    if not text.strip():
        st.warning("Please describe the incident.")
    else:
        label, confidence = classify(text)

        st.subheader(f"Result: {label}")
        st.progress(confidence)

        info = get_guidance(label)

        if "law" in info:
            st.error("⚠️ Serious Case Detected")
            st.markdown(f"**Relevant Laws (India):** {info['law']}")

            st.markdown("**Evidence Collection Tips:**")
            for tip in info["tips"]:
                st.markdown(f"- {tip}")

            st.markdown("**SOS / Emergency Contacts:**")
            for c in info["sos"]:
                st.markdown(f"- {c}")
        else:
            st.success(info["message"])

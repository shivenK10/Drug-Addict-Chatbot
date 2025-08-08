import os
import streamlit as st

st.set_page_config(page_title="Recovery Buddy 🤝", page_icon="🤝", layout="centered")

try:
    from logger import Logger
    os.makedirs("Logs", exist_ok=True)
    logger = Logger(name="AddictionChatbotUI", log_file_needed=True,
                    log_file_path="Logs/streamlit_app.log", level="DEV")
except Exception:
    logger = None

@st.cache_resource(show_spinner=True)
def load_pipelines():
    from emotion_detection_pipeline import detect_emotion
    from generation_pipeline import generate_response
    return detect_emotion, generate_response

detect_emotion, generate_response = load_pipelines()

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    show_detect_badge = st.checkbox("Show detection badge", value=True)
    show_confidence = st.checkbox("Show confidence %", value=True)
    st.markdown("---")
    if st.button("🧹 Reset chat"):
        st.session_state.clear()
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "analyses" not in st.session_state:
    st.session_state.analyses = []

st.markdown("# Recovery Buddy 🤝")
st.caption(
    "I'm here to support you like a friend—no judgment. "
    "If you feel unsafe or in crisis, please reach out to local emergency services or a trusted person nearby."
)

def format_recent_history(messages, limit=6):
    """Return a compact plain-text history for the model (friend-like style)."""
    if not messages:
        return ""
    msgs = messages[-limit:]
    lines = []
    for m in msgs:
        who = "You" if m["role"] == "user" else "Buddy"
        lines.append(f"{who}: {m['content']}")
    return "\n".join(lines)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_text = st.chat_input("Tell me what's going on. I'm listening…")
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    try:
        analysis = detect_emotion(user_text, {"history": st.session_state.messages[-10:]})
    except Exception as e:
        if logger: logger.error(f"Detection error: {e}")
        analysis = {
            "primary_emotion": "neutral",
            "confidence": 0.0,
            "addiction_context": "none",
            "recommended_approach": "standard",
        }
    st.session_state.analyses.append(analysis)

    if show_detect_badge:
        badge = f"**Detected:** {analysis['primary_emotion'].title()} · **Context:** {analysis['addiction_context']}"
        if show_confidence and "confidence" in analysis:
            badge += f" · **Confidence:** {analysis['confidence']:.0%}"
        with st.chat_message("assistant"):
            st.caption(badge)

    recent = format_recent_history(st.session_state.messages, limit=6)
    friend_tone_header = (
        "Tone: warm, supportive friend; gentle, hopeful; short paragraphs; "
        "ask one caring question; practical next step; minimal emojis.\n\n"
    )
    model_input = f"{friend_tone_header}Recent chat:\n{recent}\n\nNow the user says: {user_text}"

    try:
        reply = generate_response(model_input, analysis)
    except Exception as e:
        if logger: logger.error(f"Generation error: {e}")
        reply = "I'm here with you. Could you share a little more about how it feels right now?"

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)

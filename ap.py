"""
Streamlit Chatbot powered by Groq (model: llama-3.3-70b-versatile)
--------------------------------------------------------------------

SETUP (in VS Code terminal):

    1. Install dependencies:
         pip install streamlit groq python-dotenv

    2. Create a file named ".env" in the SAME folder as this script,
       with this single line inside it (no quotes, no spaces around =):
         GROQ_API_KEY=your_actual_groq_api_key_here

    3. Run the app:
         streamlit run streamlit.py

    Your browser will open automatically at http://localhost:8501
"""

import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# ---------------------------------------------------------------------------
# Load API key from dotenv file
# ---------------------------------------------------------------------------
load_dotenv()  # reads the .env file in the same folder and loads it into os.environ
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = "llama-3.3-70b-versatile"
SYSTEM_PROMPT = "You are a helpful, friendly assistant."

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Groq Chatbot", page_icon="💬", layout="centered")
st.title("💬 Groq Chatbot")
st.caption(f"Powered by Groq · Model: `{MODEL_NAME}`")

# ---------------------------------------------------------------------------
# Stop early with a clear message if the key is missing
# ---------------------------------------------------------------------------
if not GROQ_API_KEY:
    st.error(
        "No GROQ_API_KEY found.\n\n"
        "Create a file named `.env` in this project folder containing:\n\n"
        "```\nGROQ_API_KEY=your_actual_groq_api_key_here\n```\n\n"
        "Then restart the app."
    )
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------------------------
# Session state: keep chat history across reruns
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# Sidebar controls
with st.sidebar:
    st.subheader("Settings")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.rerun()

# ---------------------------------------------------------------------------
# Render chat history (skip the system prompt)
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
user_input = st.chat_input("Type your message...")

if user_input:
    # Show and store user's message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get and stream the assistant's reply
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_reply = ""
        try:
            stream = client.chat.completions.create(
                model=MODEL_NAME,
                messages=st.session_state.messages,
                temperature=0.7,
                max_tokens=1024,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full_reply += delta
                placeholder.markdown(full_reply + "▌")
            placeholder.markdown(full_reply)
        except Exception as e:
            full_reply = f"⚠️ Error contacting Groq API: {e}"
            placeholder.markdown(full_reply)

    st.session_state.messages.append({"role": "assistant", "content": full_reply})

"""
Local web UI for testing AgriChatbot in a browser (conversational mode —
follow-ups resolve against chat history, repeated questions come from the
semantic cache). Not part of the ADTC submission — local convenience only.
"""

import streamlit as st

from chatbot import AgriChatbot

st.set_page_config(page_title="Agri Chatbot", page_icon=None, layout="centered")
st.title("Agriculture Advisory Chatbot")
st.caption("Conversational test UI — calls AgriChatbot.ask(); follow-ups resolve against chat history")


@st.cache_resource
def load_bot(use_llm: bool) -> AgriChatbot:
    return AgriChatbot(use_llm=use_llm)


use_llm = st.checkbox("Use LLM (uncheck for retrieval-only)", value=True)
bot = load_bot(use_llm)

question = st.chat_input("Ask about any crop, e.g. \"Tell me about rice planting\"")

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question:
    q = question.strip()
    st.session_state.history.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.write(q)

    with st.chat_message("assistant"):
        warning = "THE DATA BASE DID NOT SPECIFICALLY SAY THIS SO BE WARNED OF MISINFORMATION"
        
        if use_llm:
            # Stream the LLM response
            result = bot.ask(q, stream=True)
            full_answer = ""
            placeholder = st.empty()
            
            for chunk in result["answer"]:
                full_answer += chunk
                # Clean warning from the main streaming text for better display
                display_text = full_answer.replace(warning, "").strip()
                placeholder.markdown(display_text + "▌")
            
            placeholder.markdown(full_answer.replace(warning, "").strip())
            
            if result["low_confidence"]:
                st.markdown(f":red[**{warning}**]")
        else:
            # Non-LLM path
            result = bot.ask(q)
            answer = result["answer"]
            if warning in answer:
                main_answer = answer.replace(warning, "").strip()
                st.write(main_answer)
                st.markdown(f":red[**{warning}**]")
            else:
                st.write(answer)
            full_answer = answer

        origin = "LLM" if result.get("llm_generated") else "retrieved"
        st.caption(
            f"**Origin:** {origin}  \n"
            f"**Source:** {result.get('source') or '—'}  \n"
            f"**Confidence:** {result.get('confidence', 0)}  \n"
            f"**Matched:** {result.get('matched_question') or '—'}"
            + (" · low confidence" if result.get("low_confidence") else "")
        )

    st.session_state.history.append({"role": "assistant", "content": full_answer})

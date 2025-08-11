# app.py
import streamlit as st
from rag_pipeline import (
    init_components,
    process_pdfs_in_background,
    answer_query,
    init_components as init_comp_module,
)
import os
import time
import threading

st.set_page_config(page_title="StudyMate AI", layout="wide")

# ---- Sidebar ----
st.sidebar.title("📚 StudyMate AI")
st.sidebar.markdown("Upload PDFs and chat with your documents.")

uploaded_files = st.sidebar.file_uploader("📤 Upload Documents (PDF)", type=["pdf"], accept_multiple_files=True)

if "index_status" not in st.session_state:
    st.session_state.index_status = {"status": "idle", "msg": ""}

if uploaded_files:
    st.sidebar.success(f"{len(uploaded_files)} file(s) uploaded")
    if st.sidebar.button("Process Documents"):
        # start background processing
        st.session_state.index_status = {"status": "queued", "msg": "Starting..."}
        status_ref = st.session_state.index_status

        def bg_proc(files, status_ref):
            process_pdfs_in_background(files, status_ref)
            # re-init components after processing completes
            init_comp_module(force_reload=True)

        t = threading.Thread(target=bg_proc, args=(uploaded_files, status_ref), daemon=True)
        t.start()
        st.sidebar.info("Document processing started in background. Watch the status below.")

st.sidebar.markdown("**Index status:**")
st.sidebar.write(st.session_state.index_status.get("status", "idle"))
if st.session_state.index_status.get("msg"):
    st.sidebar.write(st.session_state.index_status.get("msg"))

st.sidebar.markdown("---")
st.sidebar.subheader("🕒 Recent Conversations")
# Simple placeholder — you can expand to show real history later
if "messages" in st.session_state and st.session_state["messages"]:
    # show last 5 user messages
    user_msgs = [m for m in st.session_state["messages"] if m["role"] == "user"][-5:]
    for um in user_msgs:
        st.sidebar.write(um["content"])
else:
    st.sidebar.write("No recent conversations yet.")

# ---- Main UI ----
col1, col2 = st.columns([1, 2])
with col1:
    # replace with a valid image path or remove if not present
    if os.path.exists("welcome_image.png"):
        st.image("welcome_image.png", use_container_width=True)
    else:
        st.write("")  # spacer
with col2:
    st.markdown(
        """
        ## Welcome to **StudyMate AI**
        Your all-in-one study partner for Q&A, summaries, quizzes, and more.
        """
    )

st.markdown("---")
st.markdown("### 💬 Chat with your documents")

# initialize components in module (safe to call repeatedly)
try:
    init_components()
except Exception as e:
    st.warning(f"Warning while initializing components: {e}")

# session history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history using Streamlit chat primitives
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and isinstance(msg.get("content"), dict):
            # assistant content includes answer + sources
            st.markdown(msg["content"]["answer"])
            if msg["content"].get("sources"):
                st.markdown("**Sources:**")
                for s in msg["content"]["sources"]:
                    meta = s.get("metadata", {}) or {}
                    src_label = meta.get("source", "unknown")
                    st.markdown(f"- {src_label} (score: {s.get('score', 0):.3f})")
        else:
            st.markdown(msg["content"])

# Input area (keeps at bottom)
query = st.chat_input("Type your question here...")

if query:
    # show user message immediately
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # placeholder assistant message while processing
    with st.chat_message("assistant"):
        processing_msg = st.empty()
        processing_msg.markdown("Processing...")

    # run the query (synchronously). If you want to make this non-blocking,
    # you'd need to stream results or run in thread and update session_state.
    result = answer_query(query, return_sources=True)

    # remove placeholder and show answer
    processing_msg.empty()
    with st.chat_message("assistant"):
        if result.get("error"):
            st.error(f"Error: {result['error']}")
            st.session_state.messages.append({"role": "assistant", "content": {"answer": f"Error: {result['error']}", "sources": []}})
        else:
            # Display answer and sources
            st.markdown(result["answer"])
            if result.get("sources"):
                st.markdown("**Sources:**")
                for s in result["sources"]:
                    meta = s.get("metadata", {}) or {}
                    src_label = meta.get("source", "unknown")
                    # show small snippet link-like display
                    st.markdown(f"- {src_label} (score: {s.get('score', 0):.3f})")
            # append to history with structured content for later rendering
            st.session_state.messages.append({"role": "assistant", "content": {"answer": result["answer"], "sources": result.get("sources", [])}})

st.markdown("---")
st.caption("StudyMate AI — Retrieval-Augmented Chat for PDFs (Groq LLM)")

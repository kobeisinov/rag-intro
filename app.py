"""Streamlit UI: upload a PDF, chat with it, and inspect the retrieved chunks. Run: streamlit run app.py"""

import streamlit as st

from rag import PdfIndex

st.set_page_config(page_title="Ask your PDF", page_icon="📄")
st.title("📄 Ask your PDF")


@st.cache_resource(show_spinner="Indexing PDF...")
def get_index(pdf_bytes: bytes) -> PdfIndex:
    return PdfIndex.build(pdf_bytes)


file = st.file_uploader("Upload a PDF", type="pdf")
if not file:
    st.stop()

try:
    index = get_index(file.getvalue())
except ValueError as e:
    st.error(str(e))
    st.stop()
st.caption(f"{len(index.chunks)} chunks indexed")

# Start a fresh conversation whenever a different file is uploaded.
if st.session_state.get("file") != file.file_id:
    st.session_state.file = file.file_id
    st.session_state.messages = []


def render_sources(hits):
    with st.expander("Retrieved chunks"):
        for chunk, score in hits:
            st.markdown(f"**p. {chunk.page}** · similarity {score:.2f}")
            st.caption(chunk.text)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("hits"):
            render_sources(msg["hits"])

if question := st.chat_input("Ask something about the document"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, hits = index.answer(question)
        st.markdown(answer)
        render_sources(hits)
    st.session_state.messages.append({"role": "assistant", "content": answer, "hits": hits})

import re
import streamlit as st
from src.pipeline import RAGPipeline

st.set_page_config(
    page_title="Document RAG Chatbot",
    layout="wide"
)

st.title("Document RAG Chatbot")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()

pipeline = st.session_state.pipeline
with st.sidebar:
    st.header("Configuration")
    st.write(
        f"Model: {pipeline.generator.model}"
    )
    st.write(
        f"Indexed Chunks: {pipeline.retriever.count()}"
    )
    st.divider()
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input(
    "Ask a question about the document..."
)
if query:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_stream, docs = pipeline.query(query)
    
        full_ans = ""
        for chunk in response_stream:
            try:
                ans = chunk.choices[0].delta.content
                if ans:
                    full_ans += ans
                    clean_answer = re.sub(
                        r"<think[\s\S]*?</think>",
                        "",
                        full_ans,
                        flags=re.DOTALL
                    ).strip()
                    response_placeholder.markdown(
                        clean_answer + "▌"
                    )

            except Exception:
                pass

        full_ans = re.sub(
            r"<think[\s\S]*?</think>",
            "",
            full_ans,
            flags=re.DOTALL
        ).strip()

        response_placeholder.markdown(
            full_ans
        )

        with st.expander("Source Chunks Used"):
            for idx, doc in enumerate(docs, start=1):
                page = doc.metadata.get(
                    "page",
                    "Unknown"
                )
                st.markdown(
                    f"### Source {idx} | Page {page}"
                )
                st.write(
                    doc.page_content
                )
                st.divider()
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_ans
        }
    )
import os
import json
import re
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def clean(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\.{3,}", ".", text)
    return text.strip()

def build_vector_db():
    loader = PyMuPDFLoader(
        os.getenv("PDF_PATH")
    )
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150
    )

    chunk_docs = []
    for page_num, page in enumerate(pages, start=1):

        chunks = splitter.split_text(
            clean(page.page_content)
        )
        for idx, chunk in enumerate(chunks):
            chunk_docs.append(
               Document(
                        page_content=chunk,
                        metadata={
                            "page": page_num,
                            "chunk": idx
                        }
                )
            )
    Path("chunks").mkdir(exist_ok=True)
    with open("chunks/chunks.json", "w") as f:
        json.dump(
            [
                {
                    "id": i,
                    "text": doc.page_content
                }
                for i, doc in enumerate(chunk_docs)
            ],
            f,
            indent=2
        )

    embedding_model = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBED_MODEL")
    )

    vectordb = Chroma.from_documents(
        documents=chunk_docs,
        embedding=embedding_model,
        collection_name=os.getenv("COLLECTION_NAME"),
        persist_directory=os.getenv("CHROMA_DIR")
    )

    print(
        f"Stored {vectordb._collection.count()} vectors"
    )

if __name__ == "__main__":
    build_vector_db()
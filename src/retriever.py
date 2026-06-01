import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

class Retriever:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBED_MODEL")
        )

        self.vector_store = Chroma(
            collection_name=os.getenv("COLLECTION_NAME"),
            embedding_function=self.embedding_model,
            persist_directory=os.getenv("CHROMA_DIR")
        )

        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 6,
                "fetch_k": 20
            }
        )

    def retrieve(self, query):

        return self.retriever.invoke(query)

    def count(self):

        return self.vector_store._collection.count()
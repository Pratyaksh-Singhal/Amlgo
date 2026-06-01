from src.retriever import Retriever
from src.generator import Generator

class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()

    def query(self, question):
        docs = self.retriever.retrieve(
            question
        )
        response = self.generator.stream_ans(
            question,
            docs
        )
        return response, docs
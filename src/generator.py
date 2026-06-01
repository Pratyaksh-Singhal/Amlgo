import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class Generator:
    def __init__(self):
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.model = os.getenv("GROQ_MODEL")

        self.system_prompt = '''You are a document question answering assistant. You are given a question and a set of relevant document excerpts to answer the question.
Use only the provided excerpts to answer the question. Do not use any outside knowledge. Answer ONLY from the provided context.
Rules:
1. Use only the retrieved context.
2. Do not hallucinate.
3. If the answer is unavailable, say:
   "I don't have enough information to answer that."
4. Do not output reasoning.
5. Do not output any text other than the answer.'''

    def build_messages(self, query, docs):
        ans = []
        for d in docs:
            page = d.metadata.get(
                "page",
                d.metadata.get("source_page", "Unknown")
                )
            ans.append(
                f"[Page {page}]\n{d.page_content}"
            )

        context = "\n\n".join(ans)

        return [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content":
                f""" Context: {context}
                Question:{query}"""
            }
        ]

    def stream_ans(self, query, docs):
        messages = self.build_messages(
            query,
            docs
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=512,
            stream=True
        )
        return response
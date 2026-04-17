from groq import Groq
from pathlib import Path

def load_api_key():
    key_path = Path(__file__).resolve().parent.parent / "secrets" / "groq_api_key.txt"
    with open(key_path, "r") as f:
        return f.read().strip()

class GroqClient:
    def __init__(self):
        api_key = load_api_key()
        self.client = Groq(api_key=api_key)

    def generate_answer(self, query, retrieved_chunks):
        context = "\n\n".join([c["text"] for c in retrieved_chunks])

        prompt = f"""
        You are an AI assistant.

        Answer the question ONLY from the context.

        Context:
        {context}

        Question:
        {query}

        Answer clearly:
        """

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content
import google.generativeai as genai
import os

class GeminiClient:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-pro")

    def generate(self, query, chunks):
        context = "\n\n".join(chunk["text"] for chunk in chunks)

        prompt = f"""
Answer the question using ONLY the context below.
If the context is insufficient, say so clearly.

Context:
{context}

Question:
{query}
"""

        response = self.model.generate_content(prompt)
        return response.text.strip()

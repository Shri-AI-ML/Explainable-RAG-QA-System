import os
import sys
from google import genai

# Add the current directory to sys.path to ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from load_api_key import load_api_key

class GeminiClient:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            try:
                api_key = load_api_key()
            except Exception as e:
                print(f"[WARN] Could not load API key from file: {e}")

        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set and could not be loaded from secrets file.")

        self.client = genai.Client(api_key=api_key)

    def generate_answer(self, query, chunks):
        context = "\n\n".join(c["text"] for c in chunks[:3])

        prompt = f"""
Answer the question using ONLY the context below.
If the context is insufficient, say so clearly.

Context:
{context}

Question:
{query}

Return:
- Final answer (3-4 lines)
- Mention which evidence was used
"""

        response = self.client.models.generate_content(
            model="models/gemini-2.5-flash-lite",
            contents=prompt
        )

        return response.text.strip()

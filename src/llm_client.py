import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()


class GroqClient:
    def __init__(self):
        # 1. Try to get from environment variable (which now includes .env thanks to load_dotenv)
        api_key = os.getenv("GROQ_API_KEY")

        # 2. Fallback: Try to get from local secrets file
        if not api_key:
            try:
                base_dir = Path(__file__).resolve().parent.parent
                secret_path = base_dir / "secrets" / "groq_api_key.txt"
                
                if secret_path.exists():
                    with open(secret_path, "r", encoding="utf-8") as f:
                        api_key = f.read().strip()
            except Exception as e:
                print(f"Warning: Could not read secret file gracefully. {e}")

        # 3. If still no api_key, raise error
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables or secrets file")

        self.client = Groq(api_key=api_key)

    def generate_answer(self, query, retrieved_chunks):
        context = "\n\n".join([c["text"] for c in retrieved_chunks])

        prompt = f"""
        You are an AI assistant.

        Answer the question ONLY from the context.

        Give a concise answer with bullet points.

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
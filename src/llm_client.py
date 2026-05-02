from groq import Groq
import os


class GroqClient:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

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
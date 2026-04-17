from pathlib import Path

def load_api_key():
    key_path = Path(__file__).resolve().parent.parent / "secrets" / "groq_api_key.txt"
    with open(key_path, "r") as f:
        return f.read().strip()

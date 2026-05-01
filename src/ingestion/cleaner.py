import re

def clean_text(text: str) -> str:
    """Cleans extracted text by normalizing whitespaces and newlines."""
    if not text:
        return ""
    # Remove excessive newlines
    text = re.sub(r'\n+', '\n', text)
    # Remove excessive spaces
    text = re.sub(r' +', ' ', text)
    # Strip leading/trailing whitespaces
    return text.strip()

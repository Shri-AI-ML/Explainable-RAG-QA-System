import logging
from pypdf import PdfReader

logger = logging.getLogger(__name__)

def load_pdf(file_path: str) -> str:
    """Reads a PDF file and extracts text page by page."""
    try:
        reader = PdfReader(file_path)
        text_parts = []
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
            else:
                logger.warning(f"  [SKIP] Page {page_num + 1} in {file_path} contains no text.")
        
        return "\n".join(text_parts)
    except Exception as e:
        logger.error(f"  [ERROR] Failed to read PDF {file_path}: {e}")
        return ""

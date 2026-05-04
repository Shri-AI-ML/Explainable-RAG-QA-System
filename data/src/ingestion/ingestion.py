import os
import json
import logging
import uuid
import sys
from pathlib import Path

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.json_loader import load_json
from src.ingestion.cleaner import clean_text

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Determine paths relative to the project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "clean_data.json"

def process_file(file_path: Path) -> list | None:
    logger.info(f"[INFO] Processing file: {file_path.name}")
    
    ext = file_path.suffix.lower()
    
    if ext == ".pdf":
        raw_text = load_pdf(str(file_path))
        if not raw_text.strip():
            logger.warning(f"[SKIP] No text extracted from {file_path.name}")
            return None
            
        cleaned_text = clean_text(raw_text)
        if not cleaned_text:
            logger.warning(f"[SKIP] Text was empty after cleaning for {file_path.name}")
            return None
            
        return [{
            "doc_id": str(uuid.uuid4()),
            "text": cleaned_text
        }]
        
    elif ext == ".json":
        raw_docs = load_json(str(file_path))
        if not raw_docs:
            logger.warning(f"[SKIP] No documents extracted from {file_path.name}")
            return None
            
        cleaned_docs = []
        for doc in raw_docs:
            cleaned_text = clean_text(doc["text"])
            if cleaned_text:
                cleaned_docs.append({
                    "doc_id": doc["doc_id"],
                    "text": cleaned_text
                })
                
        if not cleaned_docs:
            logger.warning(f"[SKIP] Text was empty after cleaning for all docs in {file_path.name}")
            return None
            
        return cleaned_docs
    else:
        logger.info(f"[SKIP] Unsupported file extension: {ext}")
        return None

def main():
    logger.info("[INFO] Starting ingestion pipeline...")
    
    # Ensure directories exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    documents = []
    
    # Process all files in the raw directory
    if not RAW_DIR.exists():
        logger.error(f"[ERROR] Raw directory {RAW_DIR} does not exist.")
        return
        
    for item in RAW_DIR.iterdir():
        if item.is_file():
            docs = process_file(item)
            if docs:
                documents.extend(docs)
                
    if not documents:
        logger.warning("[INFO] No valid documents processed.")
        return
        
    # Save the unified output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
        
    logger.info(f"[DONE] Ingestion complete. Processed {len(documents)} files. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

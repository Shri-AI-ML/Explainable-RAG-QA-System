import json
import logging
import uuid

logger = logging.getLogger(__name__)

def load_json(file_path: str) -> list:
    """Reads a JSON file and extracts text fields, returning a list of dicts with doc_id and text."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        docs = []
        # Handle both list of objects and single object
        if isinstance(data, list):
            for item in data:
                text = item.get("text") or item.get("content") or item.get("body")
                if not text:
                    text = str(item)
                docs.append({
                    "doc_id": str(uuid.uuid4()),
                    "text": str(text)
                })
            return docs
        elif isinstance(data, dict):
            text = data.get("text") or data.get("content") or data.get("body")
            if not text:
                text = str(data)
            docs.append({
                "doc_id": str(uuid.uuid4()),
                "text": str(text)
            })
            return docs
        else:
            logger.warning(f"  [SKIP] JSON format not supported in {file_path}. Must be object or array of objects.")
            return []
    except json.JSONDecodeError:
        logger.error(f"  [ERROR] Invalid JSON in {file_path}.")
        return []
    except Exception as e:
        logger.error(f"  [ERROR] Failed to read JSON {file_path}: {e}")
        return []

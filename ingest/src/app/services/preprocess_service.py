# app/services/preprocess_service.py
from __future__ import annotations
import logging
from typing import List

from llama_index.core import Document

# Your existing helpers
from app.services.transform_service import filter_documents, transform_special_files

EXTENSION_TO_LANGUAGE_DEFAULT = {
    ".py": "python", ".java": "java", ".kt": "kotlin", ".go": "go",
    ".js": "javascript", ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript",
    ".rb": "ruby", ".rs": "rust", ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp",
    ".cs": "csharp", ".php": "php", ".scala": "scala", ".swift": "swift",
    ".sh": "bash", ".bash": "bash", ".zsh": "zsh",
    ".yml": "yaml", ".yaml": "yaml", ".toml": "toml", ".ini": "ini", ".cfg": "ini",
    ".sql": "sql", ".md": "markdown", ".rst": "rst", ".proto": "protobuf",
    ".gradle": "gradle", ".groovy": "groovy", ".xml": "xml", ".json": "json",
}


class PreprocessService:
    @staticmethod
    def prepare_repo_documents(raw_docs: List[Document]) -> List[Document]:
        """Filter, transform, and annotate language for repo documents."""
        logging.info(f"🔍 Filtering {len(raw_docs)} raw documents")
        filtered = filter_documents(raw_docs)
        logging.info(f"🔍 Filtered to {len(filtered)} docs")

        logging.info("🔄 Applying special-file transforms")
        transformed = transform_special_files(filtered)
        logging.info(f"🔄 Transformed → {len(transformed)} docs")

        # Language tagging (mirrors controller logic you had)
        for doc in transformed:
            md = doc.metadata
            if "language" in md:
                continue
            fp = (md.get("file_path") or "").strip()
            if not fp:
                continue

            filename = fp.split("/")[-1].lower()
            # Special cases
            if filename == "dockerfile":
                md["language"] = "dockerfile"
            elif "docker-compose" in filename and (filename.endswith(".yml") or filename.endswith(".yaml")):
                md["language"] = "yaml"
            else:
                ext = "." + fp.split(".")[-1].lower() if "." in fp else ""
                md["language"] = EXTENSION_TO_LANGUAGE_DEFAULT.get(ext, ext.lstrip(".") or filename)

            logging.debug(f"🔤 language='{md['language']}' for {fp}")

        return transformed

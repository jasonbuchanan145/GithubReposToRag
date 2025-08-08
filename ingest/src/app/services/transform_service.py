from __future__ import annotations
import logging
from typing import List
from llama_index.core import Document
from app.services.jupyter_notebook_handling import JupyterNotebookProcessor


SKIP_EXT = {
    ".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".feather",
    ".json", ".xml", ".jsonl", ".ndjson",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp", ".ico",
    ".tiff", ".tif", ".psd",
    ".mp3", ".wav", ".mp4", ".avi", ".mov", ".mkv", ".flv",
    ".zip", ".tar", ".gz", ".rar", ".7z", ".bz2",
    ".exe", ".dll", ".so", ".dylib", ".bin",
    ".log", ".dump", ".backup",
    ".db", ".sqlite", ".sqlite3",
}

SKIP_NAMES = {
    "license", "license.txt", "license.md",
    "changelog", "changelog.txt", "changelog.md",
    "authors", "authors.txt", "authors.md",
    "contributors", "contributors.txt", "contributors.md",
    "copying", "copying.txt", "copying.md",
    "notice", "notice.txt", "notice.md",
    ".gitignore", ".gitattributes", ".gitmodules",
    ".dockerignore", ".eslintignore", ".prettierignore",
}


def filter_documents(documents: List[Document]) -> List[Document]:
    out: List[Document] = []
    for doc in documents:
        path = doc.metadata.get("file_path", "")
        ext = ("." + path.split(".")[-1].lower()) if "." in path else ""
        name = path.split("/")[-1].lower()
        if ext in SKIP_EXT or name in SKIP_NAMES:
            continue
        out.append(doc)
    return out


def transform_special_files(documents: List[Document]) -> List[Document]:
    transformed: List[Document] = []
    for doc in documents:
        path = doc.metadata.get("file_path", "")
        if path.endswith(".ipynb") and JupyterNotebookProcessor is not None:
            try:
                processed = JupyterNotebookProcessor.process_notebook(path)
                transformed.append(
                    Document(text=processed, metadata={**doc.metadata, "content_type": "notebook", "is_processed": True})
                )
            except Exception:
                logging.warning("Notebook transform failed for %s; keeping raw text", path, exc_info=True)
                transformed.append(doc)
        else:
            transformed.append(doc)
    return transformed


def infer_component_kind(documents: List[Document]) -> str:
    """Heuristic classifier: 'service' vs 'standalone'"""
    has_nb = False
    has_manifest = False
    has_openapi = False
    for d in documents:
        p = d.metadata.get("file_path", "").lower()
        if p.endswith(".ipynb"):
            has_nb = True
        if p.endswith("package.json") or p.endswith("pyproject.toml") or p.endswith("pom.xml"):
            has_manifest = True
        if p.endswith("openapi.yaml") or p.endswith("openapi.yml") or p.endswith("openapi.json"):
            has_openapi = True
    if has_nb and not (has_manifest or has_openapi):
        return "standalone"
    return "service"
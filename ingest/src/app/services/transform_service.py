from __future__ import annotations
import logging
from collections import defaultdict
from pathlib import PurePosixPath
from typing import List
from llama_index.core import Document
from app.services.jupyter_notebook_handling import JupyterNotebookProcessor


SKIP_EXT = {
    ".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".feather",
    ".xml", ".jsonl", ".ndjson",  # Keep .json - it's important for configs
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp", ".ico",
    ".tiff", ".tif", ".psd",".drawio",
    ".mp3", ".wav", ".mp4", ".avi", ".mov", ".mkv", ".flv",
    ".zip", ".tar", ".gz", ".rar", ".7z", ".bz2",
    ".exe", ".dll", ".so", ".dylib", ".bin",
    ".log", ".dump", ".backup", ".drawio"
    ".db", ".sqlite", ".sqlite3",
}

# Add specific JSON files that should be skipped (data files, not config)
SKIP_JSON_PATTERNS = {
    "data.json", "test-data.json", "sample.json", "mock.json",
    "responses.json", "fixtures.json"
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

def top_directory(path: str, depth: int = 1) -> str:
    p = PurePosixPath(path or "")
    parts = [x for x in p.parts if x not in (".",)]
    return "/".join(parts[:depth]) if parts else ""

def group_nodes_by_file(nodes):
    by_file = defaultdict(list)
    for n in nodes:
        by_file[(n.metadata.get("file_path") or n.metadata.get("path") or "")].append(n)
    return by_file

def group_files_by_module(file_paths, depth: int = 1):
    by_mod = defaultdict(list)
    for fp in file_paths:
        by_mod[top_directory(fp, depth=depth)].append(fp)
    return by_mod

def filter_documents(documents: List[Document]) -> List[Document]:
    logging.info(f"🔍 Filtering {len(documents)} documents...")
    out: List[Document] = []
    skipped_count = 0

    for doc in documents:
        path = doc.metadata.get("file_path", "")
        ext = ("." + path.split(".")[-1].lower()) if "." in path else ""
        name = path.split("/")[-1].lower()

        # Special handling for JSON files - only skip data files, keep config files
        if ext == ".json" and name in SKIP_JSON_PATTERNS:
            logging.debug(f"  ❌ Skipping JSON data file {path}")
            skipped_count += 1
            continue
        elif ext in SKIP_EXT or name in SKIP_NAMES:
            logging.debug(f"  ❌ Skipping {path} (ext: {ext}, name: {name})")
            skipped_count += 1
            continue

        logging.debug(f"  ✅ Keeping {path} (ext: {ext})")
        out.append(doc)

    logging.info(f"🔍 Filter results: {len(out)} kept, {skipped_count} skipped")
    return out


def transform_special_files(documents: List[Document]) -> List[Document]:
    logging.info(f"🔄 Transforming {len(documents)} documents...")
    transformed: List[Document] = []
    notebook_count = 0

    for doc in documents:
        path = doc.metadata.get("file_path", "")
        doc_length = len(doc.text) if doc.text else 0

        if path.endswith(".ipynb") and JupyterNotebookProcessor is not None:
            notebook_count += 1
            try:
                processed = JupyterNotebookProcessor.process_notebook(path)
                processed_length = len(processed) if processed else 0
                logging.debug(f"  📓 Processed notebook {path}: {doc_length} → {processed_length} chars")
                transformed.append(
                    Document(text=processed, metadata={**doc.metadata, "content_type": "notebook", "is_processed": True})
                )
            except Exception:
                logging.warning("Notebook transform failed for %s; keeping raw text", path, exc_info=True)
                transformed.append(doc)
        else:
            logging.debug(f"  📄 Keeping {path} as-is ({doc_length} chars)")
            transformed.append(doc)

    logging.info(f"🔄 Transform results: {len(transformed)} docs ({notebook_count} notebooks processed)")
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
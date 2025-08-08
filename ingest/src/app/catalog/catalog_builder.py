from __future__ import annotations
from typing import List, Optional
from llama_index.core import Document


def make_catalog_document(
        repo: str,
        docs: List[Document],
        *,
        layer: Optional[str] = None,
        collection: Optional[str] = None,
        component_kind: Optional[str] = None,
) -> Document:
    """Create a single summary document describing a component for routing."""
    readme_texts, manifest_texts, api_texts = [], [], []
    for d in docs:
        p = d.metadata.get("file_path", "").lower()
        if p.endswith("readme.md"):
            readme_texts.append(d.text)
        if p.endswith("package.json") or p.endswith("pyproject.toml") or p.endswith("pom.xml"):
            manifest_texts.append(d.text)
        if p.endswith("openapi.yaml") or p.endswith("openapi.yml") or p.endswith("openapi.json"):
            api_texts.append(d.text)

    text_parts = []
    if readme_texts:
        text_parts.append("# README\n" + "\n\n".join(readme_texts))
    if manifest_texts:
        text_parts.append("# MANIFEST\n" + "\n\n".join(manifest_texts))
    if api_texts:
        text_parts.append("# API\n" + "\n\n".join(api_texts))

    text = "\n\n".join(text_parts).strip() or f"Component summary placeholder for {repo}."

    meta = {
        "doc_type": "catalog",
        "repo": repo,
        "layer": layer or "unspecified",
        "collection": collection,
        "component_kind": component_kind,
    }
    return Document(text=text, metadata=meta)
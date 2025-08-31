# app/services/catalog_service.py
from __future__ import annotations
import logging
from typing import List, Any

from llama_index.core import Document
from llama_index.core.schema import BaseNode
from app.catalog.catalog_builder import make_catalog_document
from app.pipelines.catalog_pipeline import build_catalog_pipeline


class CatalogService:
    @staticmethod
    def build_catalog_nodes(
            *,
            repo: str,
            documents: List[Document],
            code_nodes: List[BaseNode],
            layer: str | None,
            collection: str | None,
            component_kind: str,
            llm: Any,
    ) -> List[BaseNode]:
        """Create a single catalog Document then run through the catalog pipeline."""
        logging.info("📋 Building catalog document")
        catalog_doc = make_catalog_document(
            repo,
            documents,
            code_nodes=code_nodes,
            layer=layer,
            collection=collection,
            component_kind=component_kind,
            llm=llm,
        )
        logging.info(f"📋 Catalog doc length: {len(catalog_doc.text)} chars")

        nodes = list(build_catalog_pipeline(llm=llm).run(documents=[catalog_doc]))
        logging.info(f"📋 Catalog pipeline produced {len(nodes)} nodes")
        return nodes

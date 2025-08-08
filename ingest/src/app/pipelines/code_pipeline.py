from __future__ import annotations
import logging
from typing import List
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import SummaryExtractor, TitleExtractor, KeywordExtractor
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.schema import BaseNode, Document
from llama_index.core.llms import CustomLLM

from app.langauge_detector import create_code_splitter_safely
from app.llm_init import QwenLLM


class DynamicCodeSplitter(NodeParser):
    """Dynamic code splitter that uses document metadata to determine language."""

    def _parse_nodes(self, nodes: List[BaseNode], show_progress: bool = False) -> List[BaseNode]:
        """Required abstract method - not used in our implementation."""
        return nodes

    def get_nodes_from_documents(self, documents: List[Document], show_progress: bool = False) -> List[BaseNode]:
        """Split documents using language-specific splitters based on file metadata."""
        logging.info(f"🔧 DynamicCodeSplitter.get_nodes_from_documents called with {len(documents)} documents")

        if not documents:
            logging.error("❌ No documents received by DynamicCodeSplitter!")
            return []

        all_nodes = []

        for i, doc in enumerate(documents):
            # Extract file path and language from document metadata
            file_path = doc.metadata.get('file_path') or doc.metadata.get('path')
            language = doc.metadata.get('language')
            doc_length = len(doc.text) if doc.text else 0

            logging.info(f"📄 Document {i+1}: path='{file_path}', language='{language}', length={doc_length}")
            logging.debug(f"    Content preview: {doc.text[:100] if doc.text else 'EMPTY'}...")

            # Create appropriate splitter for this document
            splitter = create_code_splitter_safely(
                file_path=file_path,
                language=language,
                document_content=doc.text
            )

            # Split this specific document
            nodes = splitter.get_nodes_from_documents([doc], show_progress=False)
            logging.info(f"  → Generated {len(nodes)} nodes from {file_path}")
            all_nodes.extend(nodes)

        logging.info(f"✅ DynamicCodeSplitter generated {len(all_nodes)} total nodes")

        return all_nodes


def build_code_pipeline(llm: CustomLLM = None) -> IngestionPipeline:
    """Build code pipeline with dynamic language-aware splitting."""
    if llm is None:
        llm = QwenLLM()

    logging.info(f"🏗️ Building code pipeline with LLM: {type(llm).__name__}")

    return IngestionPipeline(
        transformations=[
            DynamicCodeSplitter(),
            SummaryExtractor(summaries=["self"], show_progress=True, llm=llm),
            TitleExtractor(nodes=5, llm=llm),
            KeywordExtractor(keywords=10, llm=llm),
        ]
    )
from __future__ import annotations
from typing import List
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import SummaryExtractor, TitleExtractor, KeywordExtractor
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.schema import BaseNode, Document
from llama_index.core.llms.base import BaseLLM

from app.langauge_detector import create_code_splitter_safely
from app.llm_init import QwenLLM


class DynamicCodeSplitter(NodeParser):
    """Dynamic code splitter that uses document metadata to determine language."""

    def get_nodes_from_documents(self, documents: List[Document], show_progress: bool = False) -> List[BaseNode]:
        """Split documents using language-specific splitters based on file metadata."""
        all_nodes = []

        for doc in documents:
            # Extract file path and language from document metadata
            file_path = doc.metadata.get('file_path') or doc.metadata.get('path')
            language = doc.metadata.get('language')

            # Create appropriate splitter for this document
            splitter = create_code_splitter_safely(
                file_path=file_path,
                language=language,
                document_content=doc.text
            )

            # Split this specific document
            nodes = splitter.get_nodes_from_documents([doc], show_progress=False)
            all_nodes.extend(nodes)

        return all_nodes


def build_code_pipeline(llm: BaseLLM = None) -> IngestionPipeline:
    """Build code pipeline with dynamic language-aware splitting."""
    if llm is None:
        llm = QwenLLM()

    return IngestionPipeline(
        transformations=[
            DynamicCodeSplitter(),
            SummaryExtractor(summaries=["self"], show_progress=True, llm=llm),
            TitleExtractor(nodes=5, llm=llm),
            KeywordExtractor(keywords=10, llm=llm),
        ]
    )
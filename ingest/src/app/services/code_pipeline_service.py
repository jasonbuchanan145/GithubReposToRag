# app/services/code_pipeline_service.py
from __future__ import annotations
import logging
from typing import List, Any

from llama_index.core.schema import BaseNode
from app.pipelines.code_pipeline import DynamicCodeSplitter
from llama_index.core.extractors import SummaryExtractor, TitleExtractor, KeywordExtractor


class CodePipelineService:
    @staticmethod
    def build_code_nodes(*, documents: List[Any], llm: Any) -> List[BaseNode]:
        """Split code into chunks and enrich with summaries, titles, and keywords."""
        splitter = DynamicCodeSplitter()
        split_nodes = splitter.get_nodes_from_documents(documents, show_progress=False)
        logging.info(f"🔧 Code splitter generated {len(split_nodes)} nodes")

        if not split_nodes:
            logging.error("❌ Code splitter produced 0 nodes")
            return []

        # 1) Summaries
        logging.info(f"🧠 Generating summaries for {len(split_nodes)} nodes")
        try:
            summary_extractor = SummaryExtractor(summaries=["self"], show_progress=True, llm=llm)
            summary_metadata = summary_extractor.extract(split_nodes)
            for node, md in zip(split_nodes, summary_metadata):
                node.metadata.update(md)
        except Exception as e:
            logging.exception(f"Summary extraction failed: {e}")

        # 2) Titles
        logging.info(f" Generating titles")
        try:
            title_extractor = TitleExtractor(nodes=5, llm=llm)
            title_metadata = title_extractor.extract(split_nodes)
            for node, md in zip(split_nodes, title_metadata):
                node.metadata.update(md)
        except Exception as e:
            logging.exception(f"Title extraction failed: {e}")

        # 3) Keywords
        logging.info(f"🔑 Generating keywords")
        try:
            keyword_extractor = KeywordExtractor(keywords=10, llm=llm)
            keyword_metadata = keyword_extractor.extract(split_nodes)
            for node, md in zip(split_nodes, keyword_metadata):
                node.metadata.update(md)
        except Exception as e:
            logging.exception(f"Keyword extraction failed: {e}")

        logging.info(f"✅ Code processing completed with {len(split_nodes)} enriched nodes")
        return split_nodes

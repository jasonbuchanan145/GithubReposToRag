from __future__ import annotations
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import SummaryExtractor, TitleExtractor, KeywordExtractor
from llama_index.core.node_parser import SentenceSplitter, SimpleNodeParser
from llama_index.core.llms import CustomLLM

from app.llm_init import QwenLLM


def build_catalog_pipeline(llm: CustomLLM = None) -> IngestionPipeline:
    """Build catalog pipeline with explicit LLM configuration."""
    if llm is None:
        llm = QwenLLM()

    return IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=1500, chunk_overlap=100),
            SimpleNodeParser.from_defaults(chunk_size=1500, chunk_overlap=100),
            SummaryExtractor(summaries=["self"], show_progress=True, llm=llm),
            TitleExtractor(nodes=3, llm=llm),
            KeywordExtractor(keywords=10, llm=llm),
        ]
    )
from __future__ import annotations
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import SummaryExtractor, TitleExtractor, KeywordExtractor
from llama_index.core.node_parser import SentenceSplitter


def build_catalog_pipeline() -> IngestionPipeline:
    return IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=1500, chunk_overlap=100),
            SummaryExtractor(summaries=["self"], show_progress=True),
            TitleExtractor(nodes=3),
            KeywordExtractor(keywords=10),
        ]
    )
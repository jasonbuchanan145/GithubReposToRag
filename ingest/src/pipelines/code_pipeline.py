from __future__ import annotations
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import SummaryExtractor, TitleExtractor, KeywordExtractor
from llama_index.core.node_parser import SentenceSplitter

try:
    from scripts.src.langauge_detector import create_code_splitter_safely
except Exception:  # pragma: no cover
    create_code_splitter_safely = None  # type: ignore


def _code_splitter():
    if create_code_splitter_safely is not None:
        try:
            return create_code_splitter_safely(file_path=None, language=None, document_content=None)
        except Exception:
            pass
    return SentenceSplitter(chunk_size=4000, chunk_overlap=200)


def build_code_pipeline() -> IngestionPipeline:
    return IngestionPipeline(
        transformations=[
            _code_splitter(),
            SummaryExtractor(summaries=["self"], show_progress=True),
            TitleExtractor(nodes=5),
            KeywordExtractor(keywords=10),
        ]
    )
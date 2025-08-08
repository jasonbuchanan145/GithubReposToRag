from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

from llama_index.core import Document, VectorStoreIndex

from app.catalog.catalog_builder import make_catalog_document
from app.config import SETTINGS
from app.pipelines.catalog_pipeline import build_catalog_pipeline
from app.services.cassandra_service import CassandraService
from app.services.github_service import GithubService, fetch_repositories
from app.services.transform_service import (
    filter_documents,
    transform_special_files,
    infer_component_kind,
)
from app.streaming import stream_event, stream_step


def _ensure_dump_raw_docs(repo: str, branch: str, docs: List[Document]) -> None:
    if not SETTINGS.data_dir:
        return
    import os
    os.makedirs(os.path.join(SETTINGS.data_dir, "repos", repo), exist_ok=True)
    path = os.path.join(SETTINGS.data_dir, "repos", repo, f"raw_documents_{branch}.json")
    with open(path, "w") as f:
        json.dump([d.to_dict() for d in docs], f, indent=2)


def _attach_common_metadata(nodes, *, namespace: str, repo: str, branch: str, collection: str,
                            component_kind: str, is_standalone: bool, run_id: uuid.UUID, dev_forced: bool,
                            doc_type: str) -> None:
    for n in nodes:
        md = n.metadata
        md["namespace"] = namespace
        md["repo"] = repo
        md["branch"] = branch
        md["collection"] = collection
        md["component_kind"] = component_kind
        md["is_standalone"] = is_standalone
        md["dev_forced_standalone"] = dev_forced
        md["ingest_run_id"] = str(run_id)
        md.setdefault("doc_type", doc_type)
        md.setdefault("path", md.get("file_path"))
        # Language should already be set during document preprocessing


def ingest_component(
        *,
        repo: str,
        namespace: str,
        branch: str | None = None,
        layer: Optional[str] = None,
        collection: Optional[str] = None,
        component_kind: Optional[str] = None,
        dev_force_standalone: Optional[bool] = None,
) -> Dict[str, Union[str, int, bool]]:
    """Controller: ingest one component/repo into the code index and catalog.

    Returns a dict with audit/verification stats.
    """

    branch = branch or SETTINGS.default_branch
    collection = collection or SETTINGS.default_collection

    force = SETTINGS.dev_force_standalone if dev_force_standalone is None else bool(dev_force_standalone)

    stream_event("ingest_start", {"repo": repo, "namespace": namespace, "branch": branch, "collection": collection, "force": force})

    gh = GithubService()
    cass = CassandraService()

    handles = cass.connect()
    session = handles.session
    vector_store = cass.vector_store(session)

    # 1) Load docs
    stream_step("load_docs", repo=repo, branch=branch)
    raw_docs = gh.load_repo_documents(repo, branch)
    logging.info(f"📄 Loaded {len(raw_docs)} raw documents from GitHub")

    filtered = filter_documents(raw_docs)
    logging.info(f"🔍 After filtering: {len(filtered)} documents remain")

    transformed = transform_special_files(filtered)
    logging.info(f"🔄 After transformation: {len(transformed)} documents remain")

    # Add language detection to document metadata before processing
    # Map extensions to tree-sitter language names based on tree-sitter-language-pack
    EXTENSION_TO_LANGUAGE = {
        '.js': 'javascript',
        '.jsx': 'javascript', 
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.py': 'python',
        '.java': 'java',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.cs': 'c_sharp',
        '.php': 'php',
        '.rb': 'ruby',
        '.go': 'go',
        '.rs': 'rust',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.sh': 'bash',
        '.bash': 'bash',
        '.sql': 'sql',
        '.html': 'html',
        '.htm': 'html',
        '.css': 'css',
        '.json': 'json',
        '.xml': 'xml',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.toml': 'toml',
        '.md': 'markdown',
        '.dockerfile': 'dockerfile',
    }

    for doc in transformed:
        if "language" not in doc.metadata and doc.metadata.get("file_path"):
            file_path = doc.metadata["file_path"]
            filename = file_path.split("/")[-1].lower()  # Get just the filename

            # Special case: Dockerfile (no extension)
            if filename == 'dockerfile':
                doc.metadata["language"] = 'dockerfile'
            # Special case: docker-compose files
            elif 'docker-compose' in filename and (filename.endswith('.yml') or filename.endswith('.yaml')):
                doc.metadata["language"] = 'yaml'
            # Regular extension mapping
            elif "." in file_path:
                ext = "." + file_path.split(".")[-1].lower()
                doc.metadata["language"] = EXTENSION_TO_LANGUAGE.get(ext, ext.lstrip("."))
            else:
                # No extension, use filename as language hint
                doc.metadata["language"] = filename

            logging.debug(f"🔤 Added language '{doc.metadata['language']}' for {file_path}")

    _ensure_dump_raw_docs(repo, branch, transformed)

    # 2) Determine kind
    kind = "standalone" if force else (component_kind or infer_component_kind(transformed))
    is_standalone = (kind == "standalone")

    # Create LLM instance for both pipelines
    from app.llm_init import QwenLLM
    qwen_llm = QwenLLM()

    # 3) Build code nodes from individual documents FIRST
    stream_step("build_code_nodes")
    logging.info(f"🔧 Processing {len(transformed)} documents through code pipeline")

    # Debug: log document details before pipeline
    for i, doc in enumerate(transformed[:3]):
        logging.info(f"  📄 Input Doc {i}: {doc.metadata.get('file_path', 'unknown')} - {len(doc.text)} chars")
        logging.info(f"    Content preview: {doc.text[:100]}...")

    # Apply code processing transformations directly (IngestionPipeline had issues with our custom splitter)
    logging.info(f"🔧 Processing documents through custom code pipeline")
    from app.pipelines.code_pipeline import DynamicCodeSplitter
    from llama_index.core.extractors import SummaryExtractor, TitleExtractor, KeywordExtractor

    # 1. Split documents into nodes using language-aware splitter
    splitter = DynamicCodeSplitter()
    split_nodes = splitter.get_nodes_from_documents(transformed, show_progress=False)
    logging.info(f"🔧 Code splitter generated {len(split_nodes)} nodes")

    if not split_nodes:
        logging.error("❌ Code splitter failed to generate nodes!")
        code_nodes = []
    else:
        # 2. Generate summaries for each code chunk
        logging.info(f"🔧 Generating summaries for {len(split_nodes)} nodes")
        try:
            summary_extractor = SummaryExtractor(summaries=["self"], show_progress=True, llm=qwen_llm)
            summary_metadata = summary_extractor.extract(split_nodes)

            # Apply summary metadata back to nodes
            for node, metadata in zip(split_nodes, summary_metadata):
                node.metadata.update(metadata)

            logging.info(f"✅ Summary extraction completed for {len(split_nodes)} nodes")
        except Exception as e:
            logging.error(f"❌ Summary extraction failed: {e}")

        # 3. Extract titles/topics for each chunk
        logging.info(f"🔧 Extracting titles for {len(split_nodes)} nodes")
        try:
            title_extractor = TitleExtractor(nodes=5, llm=qwen_llm)
            title_metadata = title_extractor.extract(split_nodes)

            # Apply title metadata back to nodes
            for node, metadata in zip(split_nodes, title_metadata):
                node.metadata.update(metadata)

            logging.info(f"✅ Title extraction completed for {len(split_nodes)} nodes")
        except Exception as e:
            logging.error(f"❌ Title extraction failed: {e}")

        # 4. Extract keywords for searchability
        logging.info(f"🔧 Extracting keywords for {len(split_nodes)} nodes")
        try:
            keyword_extractor = KeywordExtractor(keywords=10, llm=qwen_llm)
            keyword_metadata = keyword_extractor.extract(split_nodes)

            # Apply keyword metadata back to nodes
            for node, metadata in zip(split_nodes, keyword_metadata):
                node.metadata.update(metadata)

            logging.info(f"✅ Keyword extraction completed for {len(split_nodes)} nodes")
        except Exception as e:
            logging.error(f"❌ Keyword extraction failed: {e}")

        # All metadata has been applied to the original nodes
        code_nodes = split_nodes
        logging.info(f"✅ Code processing completed with {len(code_nodes)} enriched nodes")
    if not code_nodes:
        logging.error(f"❌ Code pipeline failed - no nodes generated from {len(transformed)} documents")
        # Log some document details for debugging
        for i, doc in enumerate(transformed[:3]):  # First 3 docs
            logging.error(f"  Doc {i}: {doc.metadata.get('file_path', 'unknown')} - {len(doc.text)} chars")
        raise RuntimeError(f"No code nodes produced for repo={repo}")

    # 4) Build catalog doc from code summaries → nodes
    stream_step("build_catalog")
    catalog_doc = make_catalog_document(
        repo, transformed, 
        code_nodes=code_nodes,  # Pass the processed code nodes
        layer=layer, 
        collection=collection, 
        component_kind=kind,
        llm=qwen_llm
    )
    catalog_doc.metadata.update({"namespace": namespace, "branch": branch, "ingest_run_id": None})
    logging.info(f"📋 Created catalog document with {len(catalog_doc.text)} characters")
    catalog_nodes = list(build_catalog_pipeline(llm=qwen_llm).run([catalog_doc]))
    logging.info(f"📋 Catalog pipeline produced {len(catalog_nodes)} nodes")

    # Generate run_id and attach metadata to both node types
    run_id = uuid.uuid4()

    # Attach metadata to code nodes (primary content)
    _attach_common_metadata(
        code_nodes,
        namespace=namespace,
        repo=repo,
        branch=branch,
        collection=collection,
        component_kind=kind,
        is_standalone=is_standalone,
        run_id=run_id,
        dev_forced=force,
        doc_type="code",
    )

    # Attach metadata to catalog nodes (stable UUID for consistency)
    _attach_common_metadata(
        catalog_nodes,
        namespace=namespace,
        repo=repo,
        branch=branch,
        collection=collection,
        component_kind=kind,
        is_standalone=is_standalone,
        run_id=uuid.UUID(int=0),  # stable run_id for catalog-only nodes
        dev_forced=force,
        doc_type="catalog",
    )

    # 5) Write & verify
    stream_step("write_vector_store", table=SETTINGS.embeddings_table)
    before_total = cass.count_rows_total(session)
    try:
        # write catalog (tiny) then code (big)
        VectorStoreIndex.from_nodes(catalog_nodes, vector_store=vector_store, show_progress=False)
        VectorStoreIndex.from_nodes(code_nodes, vector_store=vector_store, show_progress=True)
    except Exception:
        logging.exception("Cassandra write failed repo=%s ns=%s collection=%s kind=%s", repo, namespace, collection, kind)
        raise
    after_total = cass.count_rows_total(session)

    written = after_total - before_total
    if written <= 0:
        raise RuntimeError(
            f"No new rows written to {SETTINGS.cassandra_keyspace}.{SETTINGS.embeddings_table} (repo={repo}, namespace={namespace})."
        )

    # Audit
    session.execute(
        "INSERT INTO ingest_runs (run_id, namespace, repo, branch, collection, component_kind, started_at, finished_at, node_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, namespace, repo, branch, collection, kind, datetime.utcnow(), datetime.utcnow(), len(code_nodes)),
    )

    # 6) Close
    try:
        session.shutdown()
        handles.cluster.shutdown()
    except Exception:
        pass

    stream_event("ingest_done", {"repo": repo, "namespace": namespace, "written": int(written), "nodes": len(code_nodes)})

    return {
        "ok": True,
        "run_id": str(run_id),
        "repo": repo,
        "namespace": namespace,
        "collection": collection,
        "component_kind": kind,
        "branch": branch,
        "nodes_written": len(code_nodes),
        "table_rows_added": int(written),
        "is_standalone": is_standalone,
        "dev_forced_standalone": force,
    }


def ingest_many(
        components: List[Union[Tuple[str, str, Optional[str], Optional[str], Optional[str], Optional[bool]], dict]],
        *,
        branch: Optional[str] = None,
        dev_force_standalone: Optional[bool] = None,
) -> List[dict]:
    """Batch driver with flexible input formats.

    Accepts either:
      - dicts: {repo, namespace, layer?, collection?, component_kind?, branch?, dev_force_standalone?}
      - tuples: (repo, namespace, layer?, collection?, component_kind?, dev_force_standalone?)

    The top-level `dev_force_standalone` acts as a default for all items unless overridden per item.
    """
    results = []
    default_branch = branch or SETTINGS.default_branch
    if dev_force_standalone:
        for repo in fetch_repositories(SETTINGS.github_user,SETTINGS.github_token):
            params = {
                "repo": repo,
                "namespace": "default",
                "layer" : None,
                "component_kind": None,
                "branch": default_branch,
                "dev_force_standalone": True,
            }
            results.append(ingest_component(**params))
    else:
        for item in components:
            if isinstance(item, dict):
                params = dict(item)
                params.setdefault("branch", params.pop("branch", default_branch))
                if "dev_force_standalone" not in params:
                    params["dev_force_standalone"] = dev_force_standalone
            else:
                # tuple unpack (repo, namespace, layer?, collection?, component_kind?, dev_force_standalone?)
                repo = item[0]
                namespace = item[1]
                layer = item[2] if len(item) > 2 else None
                collection = item[3] if len(item) > 3 else None
                component_kind = item[4] if len(item) > 4 else None
                dev_flag = item[5] if len(item) > 5 else dev_force_standalone
                params = {
                    "repo": repo,
                    "namespace": namespace,
                    "layer": layer,
                    "collection": collection,
                    "component_kind": component_kind,
                    "branch": default_branch,
                    "dev_force_standalone": dev_flag,
                }
            results.append(ingest_component(**params))
    return results

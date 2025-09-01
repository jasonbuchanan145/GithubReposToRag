from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

# Convert UUID to proper format for Cassandra
from cassandra.util import uuid_from_time
from llama_index.core import Document
# Use IngestionPipeline to write nodes directly to vector store
from langchain_huggingface import HuggingFaceEmbeddings

import rag_shared.config
from app.config import SETTINGS
from app.llm_init import QwenLLM
from app.services.cassandra_service import CassandraService
from app.services.catalog_service import CatalogService
from app.services.code_pipeline_service import CodePipelineService
from app.services.github_service import GithubService, fetch_repositories
from app.services.hierarchy_summary_service import HierarchySummaryService
from app.services.preprocess_service import PreprocessService
from app.services.transform_service import (
    infer_component_kind,
)
from app.services.vector_write_service import VectorWriteService
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
        if doc_type == "catalog":
            md["scope"] = "catalog"
        elif doc_type == "repo":
            md["scope"] = "repo"
        elif doc_type == "module":
            md["scope"] = "module"
        elif doc_type == "file":
            md["scope"] = "file"
        else:
            md["scope"] = "chunk"
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
    """
    Controller: ingest one component/repo into hierarchical RAG (catalog, repo, module, file, chunk).

    Keeps this thin by delegating to services:
      - PreprocessService.prepare_repo_documents
      - CodePipelineService.build_code_nodes
      - CatalogService.build_catalog_nodes
      - HierarchySummaryService.build_file_nodes / build_module_nodes / build_repo_nodes
      - VectorWriteService.write_nodes_per_scope
    """

    # ── config & services ───────────────────────────────────────────────────
    branch = branch or SETTINGS.default_branch
    collection = collection or SETTINGS.default_collection
    force = SETTINGS.dev_force_standalone if dev_force_standalone is None else bool(dev_force_standalone)

    stream_event("ingest_start", {"repo": repo, "namespace": namespace, "branch": branch, "collection": collection, "force": force})

    gh   = GithubService()
    cass = CassandraService()
    cass.connect()
    qwen_llm = QwenLLM()

    handles = cass.connect()
    session = handles.session

    kind = "standalone" if force else (component_kind or infer_component_kind([]))  # real kind is set after docs load
    is_standalone = (kind == "standalone")
    run_id = uuid.uuid4()

    # ── 1) Load & normalize docs ────────────────────────────────────────────
    stream_step("load_docs", repo=repo, branch=branch)
    raw_docs = gh.load_repo_documents(repo, branch)
    stream_step("preprocess_docs", repo=repo, branch=branch)

    # delegate filtering/transform/language tagging to a service
    transformed = PreprocessService.prepare_repo_documents(raw_docs)
    # re-infer component kind now that we have content
    kind = "standalone" if force else (component_kind or infer_component_kind(transformed))
    is_standalone = (kind == "standalone")

    _ensure_dump_raw_docs(repo, branch, transformed)  # keep existing debug dump

    # ── 2) Build CHUNK (L4) nodes from code ─────────────────────────────────
    stream_step("build_code_nodes")
    code_nodes = CodePipelineService.build_code_nodes(
        documents=transformed,
        llm=qwen_llm,
    )

    # annotate shared metadata (scope/doc_type set here for consistency)
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
        doc_type="code",          # scope will be normalized to "chunk" by _attach_common_metadata
    )

    # ── 3) Build CATALOG (L0) nodes ─────────────────────────────────────────
    stream_step("build_catalog")
    catalog_nodes = CatalogService.build_catalog_nodes(
        repo=repo,
        documents=transformed,
        code_nodes=code_nodes,
        layer=layer,
        collection=collection,
        component_kind=kind,
        llm=qwen_llm,
    )
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
        doc_type="catalog",       # scope => "catalog"
    )

    # ── 4) Build FILE (L3), MODULE (L2), REPO (L1) summaries ────────────────
    stream_step("build_file_summaries")
    file_nodes = HierarchySummaryService.build_file_nodes(
        code_nodes=code_nodes,
        repo=repo,
        namespace=namespace,
        branch=branch,
        component_kind=kind,
        llm=qwen_llm,
    )
    _attach_common_metadata(
        file_nodes,
        namespace=namespace,
        repo=repo,
        branch=branch,
        collection=collection,
        component_kind=kind,
        is_standalone=is_standalone,
        run_id=run_id,
        dev_forced=force,
        doc_type="file",          # scope => "file"
    )

    stream_step("build_module_summaries")
    module_nodes = HierarchySummaryService.build_module_nodes(
        file_nodes=file_nodes,
        repo=repo,
        namespace=namespace,
        branch=branch,
        component_kind=kind,
        llm=qwen_llm,
    )
    _attach_common_metadata(
        module_nodes,
        namespace=namespace,
        repo=repo,
        branch=branch,
        collection=collection,
        component_kind=kind,
        is_standalone=is_standalone,
        run_id=run_id,
        dev_forced=force,
        doc_type="module",        # scope => "module"
    )

    stream_step("build_repo_overview")
    repo_nodes = HierarchySummaryService.build_repo_nodes(
        transformed_docs=transformed,
        module_nodes=module_nodes,
        repo=repo,
        namespace=namespace,
        branch=branch,
        component_kind=kind,
        llm=qwen_llm,
    )
    _attach_common_metadata(
        repo_nodes,
        namespace=namespace,
        repo=repo,
        branch=branch,
        collection=collection,
        component_kind=kind,
        is_standalone=is_standalone,
        run_id=run_id,
        dev_forced=force,
        doc_type="repo",          # scope => "repo"
    )

    # ── 5) Write per-scope to Cassandra vector tables ───────────────────────
    stream_step("write_vector_store_multi")
    stores = {
        "catalog": cass.vector_store(session, table=SETTINGS.embeddings_table_catalog),
        "repo":    cass.vector_store(session, table=SETTINGS.embeddings_table_repo),
        "module":  cass.vector_store(session, table=SETTINGS.embeddings_table_module),
        "file":    cass.vector_store(session, table=SETTINGS.embeddings_table_file),
        "chunk":   cass.vector_store(session, table=SETTINGS.embeddings_table_chunk),
    }
    embedder = HuggingFaceEmbeddings(model_name=rag_shared.config.EMBED_MODEL)

    VectorWriteService.write_nodes_per_scope(
        embedder=embedder,
        stores=stores,
        catalog_nodes=catalog_nodes,
        repo_nodes=repo_nodes,
        module_nodes=module_nodes,
        file_nodes=file_nodes,
        chunk_nodes=code_nodes,
    )

    # ── 6) Done ─────────────────────────────────────────────────────────────
    stream_event("ingest_done", {"repo": repo, "namespace": namespace, "branch": branch})


    # Audit with debugging and proper parameter handling
    try:
        # Debug logging to identify the problematic parameter
        logging.info(f"🔍 Audit parameters debug:")
        logging.info(f"  run_id: {run_id} (type: {type(run_id)})")
        logging.info(f"  namespace: '{namespace}' (type: {type(namespace)})")
        logging.info(f"  repo: '{repo}' (type: {type(repo)})")
        logging.info(f"  branch: '{branch}' (type: {type(branch)})")
        logging.info(f"  collection: '{collection}' (type: {type(collection)})")
        logging.info(f"  kind: '{kind}' (type: {type(kind)})")
        logging.info(f"  node_count: {len(code_nodes)} (type: {type(len(code_nodes))})")

        # Prepare parameters with proper types for Cassandra
        current_time = datetime.utcnow()
        audit_params = {
            'run_id': run_id,  # Keep as UUID object for Cassandra
            'namespace': str(namespace) if namespace else "",
            'repo': str(repo) if repo else "",
            'branch': str(branch) if branch else "",
            'collection': str(collection) if collection else "",
            'component_kind': str(kind) if kind else "",
            'started_at': current_time,
            'finished_at': current_time,
            'node_count': int(len(code_nodes))
        }

        session.execute(
            """
            INSERT INTO ingest_runs (run_id, namespace, repo, branch, collection, component_kind, started_at, finished_at, node_count) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                audit_params['run_id'],
                audit_params['namespace'], 
                audit_params['repo'],
                audit_params['branch'],
                audit_params['collection'],
                audit_params['component_kind'],
                audit_params['started_at'],
                audit_params['finished_at'],
                audit_params['node_count']
            ]
        )
        logging.info("✅ Audit record inserted successfully")

    except Exception as audit_error:
        logging.error(f"❌ Failed to insert audit record: {audit_error}")
        logging.error(f"Parameters were: {audit_params}")
        # Continue execution - don't let audit failure stop the ingestion
        logging.warning("⚠️ Continuing without audit record")

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

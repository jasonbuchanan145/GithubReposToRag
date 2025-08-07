from __future__ import annotations
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

from llama_index.core import Document, VectorStoreIndex

from config import SETTINGS
from services.github_service import GithubService, fetch_repositories
from services.cassandra_service import CassandraService
from services.transform_service import (
    filter_documents,
    transform_special_files,
    infer_component_kind,
)
from pipelines.code_pipeline import build_code_pipeline
from pipelines.catalog_pipeline import build_catalog_pipeline
from catalog.catalog_builder import make_catalog_document
from streaming import stream_event, stream_step


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
        if "language" not in md and md.get("file_path"):
            ext = ("." + md["file_path"].split(".")[-1].lower()) if "." in md.get("file_path", "") else ""
            md["language"] = ext.lstrip(".")


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
    filtered = filter_documents(raw_docs)
    transformed = transform_special_files(filtered)
    _ensure_dump_raw_docs(repo, branch, transformed)

    # 2) Determine kind
    kind = "standalone" if force else (component_kind or infer_component_kind(transformed))
    is_standalone = (kind == "standalone")

    # 3) Build catalog doc → nodes
    stream_step("build_catalog")
    catalog_doc = make_catalog_document(repo, transformed, layer=layer, collection=collection, component_kind=kind)
    catalog_doc.metadata.update({"namespace": namespace, "branch": branch, "ingest_run_id": None})
    catalog_nodes = list(build_catalog_pipeline().run([catalog_doc]))  # explicit list() for typing stability
    _attach_common_metadata(
        catalog_nodes,
        namespace=namespace,
        repo=repo,
        branch=branch,
        collection=collection,
        component_kind=kind,
        is_standalone=is_standalone,
        run_id=uuid.UUID(int=0),  # no run_id for catalog-only nodes, keep stable
        dev_forced=force,
        doc_type="catalog",
    )

    # 4) Build code nodes
    stream_step("build_code_nodes")
    code_nodes = list(build_code_pipeline().run(transformed))
    if not code_nodes:
        raise RuntimeError(f"No code nodes produced for repo={repo}")

    run_id = uuid.uuid4()
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

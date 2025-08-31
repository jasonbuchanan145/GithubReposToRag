# app/services/hierarchy_summary_service.py
from __future__ import annotations
import logging
from typing import List, Any

from llama_index.core import Document
from llama_index.core.schema import BaseNode
from app.pipelines.catalog_pipeline import build_catalog_pipeline
from app.services.scope_utils import group_nodes_by_file, group_files_by_module, top_directory


class HierarchySummaryService:
    @staticmethod
    def build_file_nodes(
            *,
            code_nodes: List[BaseNode],
            repo: str,
            namespace: str,
            branch: str,
            component_kind: str,
            llm: Any,
    ) -> List[BaseNode]:
        """Create one summary per file (roll up its chunks)."""
        files_map = group_nodes_by_file(code_nodes)
        logging.info(f"🗂 Creating file summaries for {len(files_map)} files")

        file_summary_docs: List[Document] = []

        for file_path, nodes in files_map.items():
            if not file_path:
                continue
            concat = "\n\n".join([n.get_content(metadata_mode="none") or "" for n in nodes])[:25000]
            prompt = (
                "You are creating a high-level FILE SUMMARY for developers and retrieval.\n"
                f"Path: {file_path}\n"
                "Summarize responsibilities, main APIs/entry points, external dependencies, and debugging gotchas.\n"
                "Avoid boilerplate; keep it under ~200–300 words."
            )
            try:
                text = llm.complete(prompt + "\n\n" + concat).text.strip()
            except Exception as e:
                logging.exception(f"File summary LLM failed for {file_path}: {e}")
                text = f"{file_path} summary unavailable."

            doc = Document(
                text=text,
                metadata={
                    "namespace": namespace,
                    "repo": repo,
                    "branch": branch,
                    "file_path": file_path,
                    "module": top_directory(file_path, depth=1),
                    "component_kind": component_kind,
                    "doc_type": "file",   # controller will normalize scope later
                },
            )
            file_summary_docs.append(doc)

        file_nodes = list(build_catalog_pipeline(llm=llm).run(documents=file_summary_docs))
        logging.info(f"🗂 File summary pipeline produced {len(file_nodes)} nodes")
        return file_nodes

    @staticmethod
    def build_module_nodes(
            *,
            file_nodes: List[BaseNode],
            repo: str,
            namespace: str,
            branch: str,
            component_kind: str,
            llm: Any,
            depth: int = 1,
            max_files_per_module: int = 40,
    ) -> List[BaseNode]:
        """Create one summary per top-level module/dir by aggregating file summaries."""
        # Gather file_path → summary text from file_nodes
        file_summaries = {}
        for n in file_nodes:
            fp = (n.metadata.get("file_path") or "").strip()
            if not fp:
                continue
            file_summaries[fp] = n.get_content(metadata_mode="none") or ""

        module_map = group_files_by_module(list(file_summaries.keys()), depth=depth)
        logging.info(f"📦 Creating module summaries for {len(module_map)} modules")

        module_docs: List[Document] = []
        for module, files in module_map.items():
            if not module:
                continue
            selected = files[:max_files_per_module]
            joined = "\n\n".join([file_summaries[fp] for fp in selected if fp in file_summaries])[:25000]

            prompt = (
                f"MODULE SUMMARY for '{module}' in repo {repo}.\n"
                "Aggregate responsibilities, key subcomponents, boundaries, external integrations, and ops pitfalls.\n"
                "Produce a concise overview appropriate for routing debugging and how-to questions."
            )
            try:
                text = llm.complete(prompt + "\n\n" + joined).text.strip()
            except Exception as e:
                logging.exception(f"Module summary LLM failed for {module}: {e}")
                text = f"{module} module summary unavailable."

            doc = Document(
                text=text,
                metadata={
                    "namespace": namespace,
                    "repo": repo,
                    "branch": branch,
                    "module": module,
                    "component_kind": component_kind,
                    "doc_type": "module",
                },
            )
            module_docs.append(doc)

        module_nodes = list(build_catalog_pipeline(llm=llm).run(documents=module_docs))
        logging.info(f"📦 Module summary pipeline produced {len(module_nodes)} nodes")
        return module_nodes

    @staticmethod
    def build_repo_nodes(
            *,
            transformed_docs: List[Document],
            module_nodes: List[BaseNode],
            repo: str,
            namespace: str,
            branch: str,
            component_kind: str,
            llm: Any,
            readme_limit: int = 3,
            module_limit: int = 10,
    ) -> List[BaseNode]:
        """Create a single repo overview from README/doc signals and module summaries."""
        readmes = [d.text for d in transformed_docs if (d.metadata.get("file_path", "").lower().endswith("readme.md"))]
        readmes = readmes[:readme_limit]

        module_snips = [(mn.get_content(metadata_mode="none") or "") for mn in module_nodes][:module_limit]
        seeds = "\n\n".join(readmes + module_snips)[:25000]

        prompt = (
            f"REPO OVERVIEW for {repo}:\n"
            "Provide purpose, primary services/modules, tech stack, data stores/queues, deployment/runtime, "
            "and the most common user asks. Be concise and actionable."
        )
        try:
            overview_text = llm.complete(prompt + "\n\n" + seeds).text.strip()
        except Exception as e:
            logging.exception(f"Repo overview LLM failed for {repo}: {e}")
            overview_text = f"{repo}: overview unavailable."

        repo_doc = Document(
            text=overview_text,
            metadata={
                "namespace": namespace,
                "repo": repo,
                "branch": branch,
                "component_kind": component_kind,
                "doc_type": "repo",
            },
        )
        repo_nodes = list(build_catalog_pipeline(llm=llm).run(documents=[repo_doc]))
        logging.info(f"🏷  Repo overview pipeline produced {len(repo_nodes)} nodes")
        return repo_nodes

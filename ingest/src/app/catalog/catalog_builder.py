from __future__ import annotations
import logging
from typing import List, Optional
from llama_index.core import Document
from llama_index.core.llms import CustomLLM


def _evaluate_readme_quality(readme_text: str, llm: CustomLLM) -> bool:
    """Use LLM to evaluate if README is useful for understanding the project."""
    if not readme_text or len(readme_text.strip()) < 50:
        return False

    prompt = f"""
Evaluate if this README provides useful information for understanding what this software project does.
A good README should explain the purpose, functionality, or architecture of the project.
A bad README contains only stubs, todos, boilerplate, or very minimal information.

README content:
{readme_text[:1000]}...

Respond with only "GOOD" if the README is useful for understanding the project, or "BAD" if it's just a stub/placeholder or does not provide enough information.
"""

    try:
        response = llm.complete(prompt)
        result = response.text.strip().upper()
        return result == "GOOD"
    except Exception as e:
        logging.warning(f"README quality evaluation failed: {e}")
        # Fallback to simple heuristics
        return len(readme_text.strip()) > 200 and "todo" not in readme_text.lower()


def _generate_catalog_from_code(repo: str, docs: List[Document], llm: CustomLLM) -> str:
    """Generate a catalog document from code files when README is inadequate."""
    logging.info(f"📝 Generating catalog from code analysis for {repo}")

    # Collect key files for analysis
    key_files = []
    for doc in docs:
        path = doc.metadata.get("file_path", "").lower()
        # Prioritize main files, configs, and entry points
        if any(pattern in path for pattern in [
            "main.", "index.", "app.", "__init__.py", "server.", "api.",
            "package.json", "pyproject.toml", "pom.xml", "dockerfile",
            "requirements.txt", "cargo.toml"
        ]):
            # Take first 500 chars to avoid token limits
            content_sample = doc.text[:500] if doc.text else ""
            key_files.append(f"File: {doc.metadata.get('file_path', 'unknown')}\n{content_sample}")

    # Fallback: if no key files found, use first few files
    if not key_files:
        for doc in docs[:3]:
            content_sample = doc.text[:300] if doc.text else ""
            key_files.append(f"File: {doc.metadata.get('file_path', 'unknown')}\n{content_sample}")

    files_context = "\n\n---\n\n".join(key_files[:5])  # Limit to 5 files

    prompt = f"""
Analyze this code repository and create a concise project summary that explains:
1. What this software project does (purpose/functionality)
2. Key technologies/frameworks used
3. Main components or architecture
4. How it fits into a larger system (if applicable)

Repository: {repo}
Key files:

{files_context}

Write a clear, informative summary in markdown format suitable for helping an AI agent understand what this component does and how it relates to other services.
"""

    try:
        response = llm.complete(prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Failed to generate catalog from code: {e}")
        return f"Code-based summary for {repo} (analysis failed)"


def make_catalog_document(
        repo: str,
        docs: List[Document],
        *,
        code_nodes: Optional[List] = None,
        layer: Optional[str] = None,
        collection: Optional[str] = None,
        component_kind: Optional[str] = None,
        llm: Optional[CustomLLM] = None,
) -> Document:
    """Create a single summary document describing a component for routing."""

    # First check for existing README
    readme_texts = []
    for d in docs:
        p = d.metadata.get("file_path", "").lower()
        if p.endswith("readme.md") or p.endswith("readme.txt") or p == "readme":
            readme_texts.append(d.text)

    readme_content = "\n\n".join(readme_texts) if readme_texts else ""

    # Use README if it's good quality
    if readme_content and llm and _evaluate_readme_quality(readme_content, llm):
        logging.info(f"✅ Using existing README for {repo}")
        catalog_text = f"# PROJECT OVERVIEW\n{readme_content}"

    # Otherwise, generate catalog from code-level summaries
    elif code_nodes and llm:
        logging.info(f"📝 Generating catalog from {len(code_nodes)} code summaries for {repo}")
        catalog_text = _generate_catalog_from_code_summaries(repo, code_nodes, llm)

    # Fallback to basic README or placeholder
    else:
        logging.warning(f"⚠️ No code summaries or LLM available for {repo}, using basic approach")
        if readme_content:
            catalog_text = f"# PROJECT OVERVIEW\n{readme_content}"
        else:
            catalog_text = f"Component summary placeholder for {repo}."

    meta = {
        "doc_type": "catalog",
        "repo": repo,
        "layer": layer or "unspecified",
        "collection": collection,
        "component_kind": component_kind,
        "generated_from_code_summaries": bool(code_nodes and llm),
    }
    return Document(text=catalog_text, metadata=meta)


def _generate_catalog_from_code_summaries(repo: str, code_nodes: List, llm: CustomLLM) -> str:
    """Generate architectural catalog from code-level summaries."""

    # Extract summaries from code nodes (these were created by SummaryExtractor)
    summaries = []
    file_types = set()

    for node in code_nodes:
        # Get the summary that was generated by the code pipeline
        node_summary = node.metadata.get('section_summary', '') or node.text[:200]
        file_path = node.metadata.get('file_path', 'unknown')

        if node_summary and len(node_summary.strip()) > 20:
            summaries.append(f"File: {file_path}\nSummary: {node_summary}")

        # Track file types for technology detection
        if file_path != 'unknown':
            ext = file_path.split('.')[-1].lower() if '.' in file_path else ''
            if ext:
                file_types.add(ext)

    # Limit to most relevant summaries to avoid token limits
    summary_text = "\n\n---\n\n".join(summaries[:10])
    tech_stack = ", ".join(sorted(file_types)) if file_types else "unknown"

    prompt = f"""
Based on these code-level summaries, create a comprehensive project catalog entry that explains:

1. **Purpose & Functionality**: What this software component does
2. **Architecture & Design**: Key architectural patterns and components
3. **Technology Stack**: Technologies and frameworks used
4. **Integration Points**: How it connects to other services/systems
5. **Key Features**: Main capabilities and functionality

Repository: {repo}
Detected Technologies: {tech_stack}

Code Summaries:
{summary_text}

Create a clear, structured catalog entry in markdown format that would help an AI agent understand:
- What this component is responsible for
- How it fits into a larger system architecture
- What other components might need to be updated when this changes
- Key entry points and interfaces

Focus on architectural understanding rather than implementation details.
"""

    try:
        response = llm.complete(prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Failed to generate catalog from code summaries: {e}")
        return f"# {repo}\n\nCode-based architectural summary (generation failed)\n\nDetected technologies: {tech_stack}"
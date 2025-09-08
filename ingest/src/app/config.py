from __future__ import annotations
import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


@dataclass(frozen=True)
class SettingsConfig:
    # GitHub
    github_token: str = os.environ.get("GITHUB_TOKEN", "")
    github_user: str = os.environ.get("GITHUB_USER", "jasonbuchanan145")

    # Data
    data_dir: str | None = os.environ.get("DATA_DIR")

    # Cassandra
    cassandra_host: str = os.environ.get("CASSANDRA_HOST", "localhost")
    cassandra_port: int = int(os.environ.get("CASSANDRA_PORT", "9042"))
    cassandra_username: str = os.environ.get("CASSANDRA_USERNAME", "cassandra")
    cassandra_password: str = os.environ.get("CASSANDRA_PASSWORD", "cassandra")
    cassandra_keyspace: str = os.environ.get("CASSANDRA_KEYSPACE", "vector_store")
    embeddings_table_catalog: str = os.environ.get("EMBEDDINGS_TABLE_CATALOG", "embeddings_catalog")
    embeddings_table_repo:    str = os.environ.get("EMBEDDINGS_TABLE_REPO",    "embeddings_repo")
    embeddings_table_module:  str = os.environ.get("EMBEDDINGS_TABLE_MODULE",  "embeddings_module")
    embeddings_table_file:    str = os.environ.get("EMBEDDINGS_TABLE_FILE",    "embeddings_file")
    embeddings_table_chunk:   str = os.environ.get("EMBEDDINGS_TABLE_CHUNK",   os.environ.get("EMBEDDINGS_TABLE", "embeddings"))

    default_branch: str = os.environ.get("DEFAULT_BRANCH", "main")

    # Qwen LLM Configuration
    qwen_endpoint: str = os.environ.get("QWEN_ENDPOINT", "http://qwen:8000")
#    embed_model: str = os.environ.get("EMBED_MODEL", "intfloat/e5-small-v2")

    # Logical grouping
    default_collection: str = os.environ.get("DEFAULT_COLLECTION", "misc")

    # DEV mode flag to force standalone classification
    dev_force_standalone: bool = _env_bool("DEV_MODE", False)


SETTINGS = SettingsConfig()

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

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
    embeddings_table: str = os.environ.get("EMBEDDINGS_TABLE", "embeddings")

    # LlamaIndex
    embed_dim: int = int(os.environ.get("EMBED_DIM", "384"))
    default_branch: str = os.environ.get("DEFAULT_BRANCH", "main")

    # Logical grouping
    default_collection: str = os.environ.get("DEFAULT_COLLECTION", "misc")

    # DEV mode flag to force standalone classification
    dev_force_standalone: bool = _env_bool("DEV_FORCE_STANDALONE", False)


SETTINGS = SettingsConfig()

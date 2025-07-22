#!/usr/bin/env python
"""Clone every repo under a GitHub account/org, chunk files, embed, and write to Cassandra."""
import os, pathlib, itertools, textwrap, subprocess, tempfile, requests
from base64 import b64encode
from typing import Iterable

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from sentence_transformers import SentenceTransformer  # e5-small-v2

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/e5-small-v2")
CHUNK_SIZE = 1024 * 4  # bytes

# Get Cassandra connection details from environment variables
CASSANDRA_HOST = os.getenv("CASSANDRA_HOST", "cassandra")
CASSANDRA_PORT = int(os.getenv("CASSANDRA_PORT", "9042"))
CASSANDRA_USERNAME = os.getenv("CASSANDRA_USERNAME", "cassandra")
CASSANDRA_PASSWORD = os.getenv("CASSANDRA_PASSWORD", "testyMcTesterson")
CASSANDRA_KEYSPACE = os.getenv("CASSANDRA_KEYSPACE", "vector_store")

# Debug connection details
print(f"Connecting to Cassandra with credentials: {CASSANDRA_USERNAME}:{CASSANDRA_PASSWORD}")

# Configure authentication provider
auth_provider = PlainTextAuthProvider(username=CASSANDRA_USERNAME, password=CASSANDRA_PASSWORD)

# Connect to Cassandra without specifying keyspace first
cluster = Cluster(
    [CASSANDRA_HOST], 
    port=CASSANDRA_PORT,
    auth_provider=auth_provider
)

# Create a session without keyspace to setup schema if needed
setup_session = cluster.connect()

# Create keyspace if it doesn't exist
print(f"Ensuring keyspace {CASSANDRA_KEYSPACE} exists...")
setup_session.execute(
    f"""CREATE KEYSPACE IF NOT EXISTS {CASSANDRA_KEYSPACE} 
    WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}};"""
)

# Create table if it doesn't exist
print("Creating embeddings table if it doesn't exist...")
setup_session.execute(
    f"""CREATE TABLE IF NOT EXISTS {CASSANDRA_KEYSPACE}.embeddings (
        id TEXT PRIMARY KEY,
        content TEXT,
        embedding BLOB,
        metadata MAP<TEXT, TEXT>
    );"""
)

# Now connect with the keyspace
session = cluster.connect(CASSANDRA_KEYSPACE)

# Verify connection
print(f"Connected to Cassandra at {CASSANDRA_HOST}:{CASSANDRA_PORT} using keyspace {CASSANDRA_KEYSPACE}")
model = SentenceTransformer(EMBED_MODEL)


def iter_repo_files(repo_url: str) -> Iterable[pathlib.Path]:
    """Yield path objects for every text/code file in the repo."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "--depth", "1", repo_url, tmp], check=True)
        root = pathlib.Path(tmp)
        for p in root.rglob("*.*"):  # refine filter as needed
            if p.is_file() and p.stat().st_size > 0:
                yield p


def chunk_text(text: str, source: str) -> Iterable[tuple[int, str]]:
    """Very simple byte-size splitter preserving line breaks."""
    current, cid = [], 0
    for line in text.splitlines(keepends=True):
        current.append(line)
        if sum(len(l) for l in current) >= CHUNK_SIZE:
            yield cid, "".join(current)
            current, cid = [], cid + 1
    if current:
        yield cid, "".join(current)


def ingest_repo(repo_url: str):
    repo_name = repo_url.rstrip("/").split("/")[-1]

    # Temporary directory path from the cloning operation
    temp_dir = None

    # Clone the repository and get the temp directory path
    for path in iter_repo_files(repo_url):
        if temp_dir is None:
            # Extract the temp directory path from the first file
            temp_dir = str(path.parent.parent)

        with open(path, "r", errors="ignore") as fh:
            text = fh.read()
        for cid, chunk in chunk_text(text, path.as_posix()):
            vec = model.encode(chunk, normalize_embeddings=True)
            session.execute(
                """INSERT INTO embeddings (id, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s)""",
                (
                    f"{repo_name}:{path.as_posix()}:{cid}",  # Create a unique ID
                    chunk,
                    b64encode(bytes(vec)).decode('utf-8'),  # Store as base64-encoded blob
                    {
                        "repo": repo_name,
                        "path": path.as_posix(),
                        "chunk_id": str(cid),
                        "content_type": "code_chunk"
                    }  # Store metadata as a map
                ),
            )

    # If we found a temporary directory, generate hierarchical summaries
    if temp_dir:
        try:
            from repo_summarizer import RepoSummarizer
            summarizer = RepoSummarizer(temp_dir, repo_name)
            summary_data = summarizer.summarize_repository()

            # Store repository-level summary
            if summary_data["summary"]:
                repo_vec = model.encode(summary_data["summary"], normalize_embeddings=True)
                session.execute(
                    """INSERT INTO embeddings (id, content, embedding, metadata)
                        VALUES (%s, %s, %s, %s)""",
                    (
                        f"{repo_name}:summary",  # Create a unique ID for the repo summary
                        summary_data["summary"],
                        b64encode(bytes(repo_vec)).decode('utf-8'),
                        {
                            "repo": repo_name,
                            "content_type": "repository_summary"
                        }
                    ),
                )

            # Store directory-level summaries
            for dir_info in summary_data["directories"]:
                if dir_info["summary"]:
                    dir_path = dir_info["path"]
                    dir_name = os.path.basename(dir_path)
                    dir_vec = model.encode(dir_info["summary"], normalize_embeddings=True)
                    session.execute(
                        """INSERT INTO embeddings (id, content, embedding, metadata)
                            VALUES (%s, %s, %s, %s)""",
                        (
                            f"{repo_name}:dir:{dir_name}",
                            dir_info["summary"],
                            b64encode(bytes(dir_vec)).decode('utf-8'),
                            {
                                "repo": repo_name,
                                "directory": dir_name,
                                "content_type": "directory_summary"
                            }
                        ),
                    )

            # Store file-level summaries
            for file_info in summary_data["files"]:
                if file_info["summary"]:
                    file_path = file_info["path"]
                    file_name = os.path.basename(file_path)
                    file_vec = model.encode(file_info["summary"], normalize_embeddings=True)
                    session.execute(
                        """INSERT INTO embeddings (id, content, embedding, metadata)
                            VALUES (%s, %s, %s, %s)""",
                        (
                            f"{repo_name}:file:{file_path}",
                            file_info["summary"],
                            b64encode(bytes(file_vec)).decode('utf-8'),
                            {
                                "repo": repo_name,
                                "path": file_path,
                                "content_type": "file_summary"
                            }
                        ),
                    )

            print(f"✅ Generated and stored hierarchical summaries for {repo_name}")
        except Exception as e:
            print(f"⚠️ Error generating summaries for {repo_name}: {e}")

    def fetch_repos(login: str):
        after = None
        headers = {"Authorization": f"Bearer {GH_TOKEN}"}
        while True:
            payload = {"query": query, "variables": {"login": login, "after": after}}
            resp = requests.post("https://api.github.com/graphql", json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()["data"]["user"]["repositories"]
            for node in data["nodes"]:
                if not node["isArchived"]:
                    yield node["cloneUrl"]
            if not data["pageInfo"]["hasNextPage"]:
                break
            after = data["pageInfo"]["endCursor"]

    repos = list(fetch_repos(GITHUB_USER))

    for r in repos:
        ingest_repo(r)

if __name__ == "__main__":
    # --- GitHub repo discovery via GraphQL ---
    GITHUB_USER = "jasonbuchanan145"
    GH_TOKEN = os.getenv("GITHUB_TOKEN")
    if not GH_TOKEN:
        raise SystemExit("GITHUB_TOKEN env var required for GraphQL API")

    query = """
    query($login: String!, $after: String) {
      user(login: $login) {
        repositories(first: 100, after: $after, isFork: false, ownerAffiliations: OWNER) {
          pageInfo { endCursor hasNextPage }
          nodes {
            cloneUrl
            isArchived
          }
        }
      }
    }
    """

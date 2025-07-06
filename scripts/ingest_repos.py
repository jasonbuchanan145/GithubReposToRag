#!/usr/bin/env python
"""Clone every repo under a GitHub account/org, chunk files, embed, and write to Cassandra."""
import os, pathlib, itertools, textwrap, subprocess, tempfile, requests
from base64 import b64encode
from typing import Iterable

from cassandra.cluster import Cluster
from sentence_transformers import SentenceTransformer  # e5-small-v2

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/e5-small-v2")
CHUNK_SIZE   = 1024 * 4  # bytes

cluster = Cluster(["cassandra"], port=9042)
session = cluster.connect("rag")
model   = SentenceTransformer(EMBED_MODEL)


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
    for path in iter_repo_files(repo_url):
        with open(path, "r", errors="ignore") as fh:
            text = fh.read()
        for cid, chunk in chunk_text(text, path.as_posix()):
            vec = model.encode(chunk, normalize_embeddings=True)
            session.execute(
                """INSERT INTO rag.chunks (repo, file_path, chunk_id, content, embedding)
                    VALUES (%s, %s, %s, %s, %s)""",
                (
                    repo_name,
                    path.as_posix(),
                    cid,
                    chunk,
                    list(vec)  # Cassandra driver converts list<float> to VECTOR
                ),
            )

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
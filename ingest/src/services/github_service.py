from __future__ import annotations
from typing import List

import requests
from llama_index.core import Document
from llama_index.readers.github import GithubClient, GithubRepositoryReader
from .config import SETTINGS
import logging

class GithubService:
    def __init__(self, token: str | None = None, owner: str | None = None):
        self._token = token or SETTINGS.github_token
        self._owner = owner or SETTINGS.github_user
        self._client = GithubClient(github_token=self._token)

    def load_repo_documents(self, repo: str, branch: str) -> List[Document]:
        reader = GithubRepositoryReader(
            github_client=self._client,
            owner=self._owner,
            repo=repo,
            verbose=False,
            concurrent_requests=6,
            timeout=60,
        )
        return reader.load_data(branch=branch)


def fetch_repositories(username: str, github_token:str) -> List[str]:
    """Fetch repositories for a GitHub user using GraphQL API."""
    logging.info(f"🔍 Fetching repositories for user: {username}")

    query = """
    query($login: String!, $after: String) {
      user(login: $login) {
        repositories(first: 100, after: $after, isFork: false, ownerAffiliations: OWNER) {
          pageInfo { endCursor hasNextPage }
          nodes {
            name
            url
            isArchived
            isPrivate
          }
        }
      }
    }
    """

    headers = {"Authorization": f"Bearer {github_token}"}
    repositories = []
    after = None

    while True:
        payload = {"query": query, "variables": {"login": username, "after": after}}
        response = requests.post(
            "https://api.github.com/graphql",
            json=payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()["data"]["user"]["repositories"]

        for node in data["nodes"]:
            # Skip archived and private repositories
            if node["isArchived"]:
                logging.info(f"⏩ Skipping archived repository: {node['name']}")
            elif node["isPrivate"]:
                logging.info(f"🔒 Skipping private repository: {node['name']}")
            else:
                # Just return the repository name since GithubRepositoryReader uses owner/repo separately
                repositories.append(node["name"])

        if not data["pageInfo"]["hasNextPage"]:
            break

        after = data["pageInfo"]["endCursor"]

    logging.info(f"📚 Found {len(repositories)} repositories")
    return repositories
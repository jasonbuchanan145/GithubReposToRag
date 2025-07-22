#!/usr/bin/env python
"""Generate hierarchical summaries for code repositories."""
import os
import pathlib
import requests
import json
import logging
from collections import defaultdict
from typing import Dict, List, Any

# Configure logging if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# Get LLM API endpoint from environment variables
LLM_ENDPOINT = os.getenv("QWEN_ENDPOINT", "http://qwen:8000")

def generate_summary(text: str, prompt_template: str) -> str:
    """Generate a summary using the LLM API."""
    prompt = prompt_template.format(content=text)

    response = requests.post(
        f"{LLM_ENDPOINT}/v1/completions",
        headers={"Content-Type": "application/json"},
        json={
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.3  # Lower temperature for more focused summaries
        },
        timeout=30
    )

    if response.status_code != 200:
        return "Error generating summary"

    result = response.json()
    return result['choices'][0]['text'] if 'choices' in result else ""


class RepoSummarizer:
    """Generate hierarchical summaries for a code repository."""

    # Prompt templates for different summary levels
    FILE_SUMMARY_TEMPLATE = """You are an expert software developer tasked with summarizing source code files.
    Below is the content of a source code file. Create a concise summary (maximum 200 words) that explains:
    1. The main purpose of this file
    2. Key functions/classes it contains
    3. How it might integrate with other parts of a codebase

    Source code:
    ```
    {content}
    ```

    Summary:"""

    DIRECTORY_SUMMARY_TEMPLATE = """You are an expert software architect tasked with summarizing code directories.
    Below are summaries of files in a directory. Create a concise summary (maximum 300 words) that explains:
    1. The overall purpose of this directory/module
    2. How the components work together
    3. The role this module likely plays in the larger project

    File summaries:
    {content}

    Directory summary:"""

    REPO_SUMMARY_TEMPLATE = """You are an expert software architect tasked with summarizing an entire code repository.
    Below are summaries of the main directories/modules in the repository. Create a comprehensive summary (maximum 500 words) that explains:
    1. The overall purpose and functionality of this project
    2. The architecture and how major components interact
    3. Key technologies, frameworks, or languages used
    4. How someone might use or contribute to this project

    Directory summaries:
    {content}

    Repository summary:"""

    def __init__(self, repo_path: str, repo_name: str):
        self.repo_path = pathlib.Path(repo_path)
        self.repo_name = repo_name
        self.file_summaries = {}
        self.dir_summaries = defaultdict(list)
        self.repository_summary = ""

    def summarize_file(self, file_path: pathlib.Path) -> str:
        """Generate a summary for a single file."""
        if str(file_path) in self.file_summaries:
            return self.file_summaries[str(file_path)]

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            logging.debug(f"Summarizing file: {file_path.name}")
            summary = generate_summary(content, self.FILE_SUMMARY_TEMPLATE)
            self.file_summaries[str(file_path)] = summary
            return summary
        except Exception as e:
            logging.warning(f"Error summarizing file {file_path}: {e}")
            return ""

    def summarize_directory(self, dir_path: pathlib.Path) -> str:
        """Generate a summary for a directory based on its file summaries."""
        # Get all file summaries in this directory
        file_summaries = []
        for file_path in dir_path.glob("*.*"):
            if file_path.is_file() and file_path.suffix in [".py", ".js", ".java", ".ts", ".c", ".cpp", ".h", ".cs", ".go", ".rb"]:
                summary = self.summarize_file(file_path)
                if summary:
                    file_summaries.append(f"File: {file_path.name}\n{summary}")

        if not file_summaries:
            return ""

        # Join file summaries and generate directory summary
        content = "\n\n".join(file_summaries)
        return generate_summary(content, self.DIRECTORY_SUMMARY_TEMPLATE)

    def summarize_repository(self) -> Dict[str, Any]:
        """Generate hierarchical summaries for the entire repository."""
        logging.info(f"📊 Generating summaries for repository: {self.repo_name}")

        # First, summarize all top-level directories
        dir_summaries = []
        dirs_to_process = []
        for item in self.repo_path.glob("*/"):
            if item.is_dir() and not item.name.startswith('.') and item.name not in ['node_modules', 'venv', '__pycache__']:
                dirs_to_process.append(item)

        logging.info(f"📂 Found {len(dirs_to_process)} top-level directories to summarize")

        for i, item in enumerate(dirs_to_process):
            logging.info(f"📂 Summarizing directory {i+1}/{len(dirs_to_process)}: {item.name}")
            dir_summary = self.summarize_directory(item)
            if dir_summary:
                dir_summaries.append(f"Directory: {item.name}\n{dir_summary}")
                self.dir_summaries[str(item)] = dir_summary

        # Generate repository-level summary
        if dir_summaries:
            logging.info(f"📝 Generating repository-level summary")
            content = "\n\n".join(dir_summaries)
            self.repository_summary = generate_summary(content, self.REPO_SUMMARY_TEMPLATE)

        # Count summaries generated
        logging.info(f"✅ Summary generation complete: {len(self.dir_summaries)} directories, {len(self.file_summaries)} files")

        # Prepare the hierarchical summary structure
        return {
            "repo_name": self.repo_name,
            "summary": self.repository_summary,
            "directories": [{
                "path": dir_path,
                "summary": summary
            } for dir_path, summary in self.dir_summaries.items()],
            "files": [{
                "path": file_path,
                "summary": summary
            } for file_path, summary in self.file_summaries.items()]
        }


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python repo_summarizer.py <repo_path> [repo_name]")
        sys.exit(1)

    repo_path = sys.argv[1]
    repo_name = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(os.path.normpath(repo_path))

    summarizer = RepoSummarizer(repo_path, repo_name)
    summary = summarizer.summarize_repository()

    print(json.dumps(summary, indent=2))

import logging
import re
from typing import List, Dict

# Jupyter notebook processing
import nbformat
from cassandra.auth import PlainTextAuthProvider
# Ensure Cassandra keyspace exists
from cassandra.cluster import Cluster
# LlamaIndex imports
from llama_index.core import Settings
# Remove the problematic StreamingResponse import - we'll create our own simple implementation
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from nbconvert.filters import strip_ansi


# Notebook processing utilities
class JupyterNotebookProcessor:
    """Process Jupyter notebooks to extract meaningful content and filter out noise."""

    # Common patterns for dependency management cells
    DEPENDENCY_PATTERNS = [
        r'^!pip install',
        r'^!conda install',
        r'^!apt-get',
        r'^!apt install',
        r'^!yum install',
        r'^%pip install',
        r'^%conda install',
        r'^import sys\s*\n\s*!\{sys\.executable\}\s+-m\s+pip\s+install'
    ]

    # Patterns for logging and debugging outputs
    LOG_PATTERNS = [
        r'^print\(.*\)',
        r'^logging\.\w+\(',
        r'^warnings\.\w+\('
    ]

    # File-system related operations often used for setup
    FILESYSTEM_PATTERNS = [
        r'^!mkdir',
        r'^!cp',
        r'^!mv',
        r'^!rm',
        r'^!wget',
        r'^!curl'
    ]

    # Other common notebook "noise"
    NOISE_PATTERNS = [
        r'^%matplotlib inline',
        r'^%config',
        r'^%load_ext',
        r'^%env',
        r'^!kaggle',
        r'^!jupyter',
        r'^!python -m'
    ]

    @classmethod
    def is_setup_cell(cls, cell_source: str) -> bool:
        """Determine if a cell is primarily for setup/configuration rather than content."""
        # Combine all patterns
        all_patterns = cls.DEPENDENCY_PATTERNS + cls.FILESYSTEM_PATTERNS + cls.NOISE_PATTERNS

        # Check if any pattern matches at the beginning of any line
        lines = cell_source.split('\n')
        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            for pattern in all_patterns:
                if re.match(pattern, line):
                    return True

        return False

    @classmethod
    def is_output_heavy(cls, cell_output: List[Dict]) -> bool:
        """Determine if cell output is primarily logs/data dumps rather than meaningful results."""
        if not cell_output:
            return False

        # Check for common output types that might be noise
        text_output = ""
        for output in cell_output:
            # Get text output if available
            if output.get('output_type') == 'stream':
                text_output += output.get('text', '')
            elif output.get('output_type') == 'execute_result':
                if 'text/plain' in output.get('data', {}):
                    text_output += output['data']['text/plain']

        # Strip ANSI escape sequences
        text_output = strip_ansi(text_output)

        # Check if output is very long (likely a data dump)
        if len(text_output) > 500:  # Arbitrary threshold
            # But allow if it contains actual results (tables, summaries, etc.)
            if '===' in text_output or '---' in text_output or '|' in text_output:
                return False
            return True

        # Check for log-like patterns
        log_patterns = [
            r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}',  # Datetime pattern
            r'DEBUG|INFO|WARNING|ERROR|CRITICAL',
            r'Downloading|Downloaded',
            r'\d+%\|[█▉▊▋▌▍▎▏ ]+\|'
        ]

        for pattern in log_patterns:
            if re.search(pattern, text_output):
                # If more than 30% of lines match log patterns, consider it log-heavy
                log_lines = sum(1 for line in text_output.split('\n') if re.search(pattern, line))
                total_lines = len(text_output.split('\n'))
                if total_lines > 0 and log_lines / total_lines > 0.3:
                    return True

        return False

    @classmethod
    def process_notebook(cls, notebook_path: str) -> str:
        """Process a Jupyter notebook to extract meaningful content."""
        try:
            # Load the notebook
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)

            # Initialize processed content
            meaningful_cells = []

            # Add notebook metadata if available
            if 'metadata' in nb and nb['metadata']:
                title = nb['metadata'].get('title', '')
                if title:
                    meaningful_cells.append(f"# {title}\n")

            # Process each cell
            for cell in nb.cells:
                # Skip empty cells
                if not cell.get('source', '').strip():
                    continue

                # Handle different cell types
                if cell['cell_type'] == 'markdown':
                    # Always include markdown as it usually contains explanations
                    meaningful_cells.append(cell['source'])

                elif cell['cell_type'] == 'code':
                    # Skip setup/configuration cells
                    if cls.is_setup_cell(cell['source']):
                        continue

                    # Check if outputs are meaningful
                    include_output = False
                    if 'outputs' in cell and cell['outputs']:
                        if not cls.is_output_heavy(cell['outputs']):
                            include_output = True

                    # Always include the code
                    meaningful_cells.append(f"```python\n{cell['source']}\n```")

                    # Include meaningful outputs
                    if include_output:
                        output_text = ""
                        for output in cell['outputs']:
                            if output.get('output_type') == 'stream':
                                output_text += output.get('text', '')
                            elif output.get('output_type') == 'execute_result':
                                if 'text/plain' in output.get('data', {}):
                                    output_text += output['data']['text/plain']

                        if output_text.strip():
                            # Strip ANSI escape sequences
                            output_text = strip_ansi(output_text)
                            meaningful_cells.append(f"```\n{output_text}\n```")

            # Combine all meaningful content
            processed_content = "\n\n".join(meaningful_cells)
            return processed_content

        except Exception as e:
            logging.warning(f"Error processing notebook {notebook_path}: {e}")
            # Return the error message but also attempt to read the file directly as fallback
            try:
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except:
                return f"Error processing notebook: {str(e)}"

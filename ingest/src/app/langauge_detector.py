import logging
import logging
from pathlib import Path
from typing import Optional

def detect_language_from_extension(file_path):
    """Detect programming language from file extension."""
    extension_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.cs': 'c_sharp',
        '.php': 'php',
        '.rb': 'ruby',
        '.go': 'go',
        '.rs': 'rust',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.sh': 'bash',
        '.sql': 'sql',
        '.html': 'html',
        '.css': 'css',
        '.json': 'json',
        '.xml': 'xml',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.md': 'markdown',
        '.dockerfile': 'dockerfile',
        # Jupyter notebooks are typically Python-based
        '.ipynb': 'python'
    }

    ext = Path(file_path).suffix.lower()
    return extension_map.get(ext, None)

def detect_notebook_kernel_language(notebook_content: str) -> Optional[str]:
    """Detect the primary language used in a Jupyter notebook from its content."""
    try:
        import json

        # Try to parse as JSON to get kernel info
        nb_data = json.loads(notebook_content)

        # Check kernel metadata
        if 'metadata' in nb_data and 'kernelspec' in nb_data['metadata']:
            kernel_name = nb_data['metadata']['kernelspec'].get('name', '').lower()
            kernel_language = nb_data['metadata']['kernelspec'].get('language', '').lower()

            # Map common kernel names to languages
            kernel_map = {
                'python3': 'python',
                'python2': 'python',
                'ir': 'r',
                'scala': 'scala',
                'julia': 'julia',
                'javascript': 'javascript',
                'typescript': 'typescript'
            }

            if kernel_name in kernel_map:
                return kernel_map[kernel_name]

            if kernel_language in ['python', 'r', 'scala', 'julia', 'javascript']:
                return kernel_language

        # Default to Python for notebooks if we can't determine
        return 'python'

    except:
        # If we can't parse the notebook, assume Python
        return 'python'

def create_code_splitter_safely(file_path=None, language=None, document_content=None):
    """Create CodeSplitter with proper language detection and fallback."""
    try:
        from llama_index.core.node_parser import CodeSplitter

        # If language is "auto" or not specified, try to detect
        if language == "auto" or language is None:
            if file_path:
                detected_language = detect_language_from_extension(file_path)

                # Special handling for Jupyter notebooks
                if file_path.endswith('.ipynb') and document_content:
                    detected_language = detect_notebook_kernel_language(document_content)

                if detected_language:
                    language = detected_language
                else:
                    # Fallback to generic text splitter for unknown file types
                    from llama_index.core.node_parser import SentenceSplitter
                    return SentenceSplitter(
                        chunk_size=4000,
                        chunk_overlap=200
                    )
            else:
                # No file path available, use generic text splitter
                from llama_index.core.node_parser import SentenceSplitter
                return SentenceSplitter(
                    chunk_size=4000,
                    chunk_overlap=200
                )

        # Try to create CodeSplitter with the determined language
        code_splitter = CodeSplitter(
            language=language,
            chunk_lines=100,  # Reasonable default for code chunks
            chunk_lines_overlap=10,
            max_chars=4000
        )
        return code_splitter

    except LookupError as e:
        # If the specific language is not supported, fallback to text splitter
        logging.warning(f"Language '{language}' not supported by tree-sitter, falling back to SentenceSplitter: {e}")
        from llama_index.core.node_parser import SentenceSplitter
        return SentenceSplitter(
            chunk_size=4000,
            chunk_overlap=200
        )
    except ImportError as e:
        logging.warning(f"CodeSplitter unavailable due to missing dependency: {e}")
        from llama_index.core.node_parser import SentenceSplitter
        return SentenceSplitter(
            chunk_size=4000,
            chunk_overlap=200
        )
    except Exception as e:
        # Any other error, use generic text splitter
        logging.warning(f"Error creating CodeSplitter, falling back to SentenceSplitter: {e}")
        from llama_index.core.node_parser import SentenceSplitter
        return SentenceSplitter(
            chunk_size=4000,
            chunk_overlap=200
        )
import os
import requests
import re
from fastapi import FastAPI
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import SimpleStatement
from sentence_transformers import SentenceTransformer

# Import system prompts
from system_prompts import INITIAL_PROMPT, FOLLOWUP_SYSTEM_PROMPT

app = FastAPI()

# Get Qwen API endpoint from environment variable
QWEN_ENDPOINT = os.getenv("QWEN_ENDPOINT", "http://qwen:8000")

# Use the same embedding model as ingest_repos.py
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/e5-small-v2")
embedding_model = SentenceTransformer(EMBED_MODEL)

# Max iterations for follow-up RAG queries
MAX_ITERATIONS = 3

# Get Cassandra connection details from environment variables
CASSANDRA_HOST = os.getenv("CASSANDRA_HOST", "cassandra")
CASSANDRA_PORT = int(os.getenv("CASSANDRA_PORT", "9042"))
CASSANDRA_USERNAME = os.getenv("CASSANDRA_USERNAME", "cassandra")
CASSANDRA_PASSWORD = os.getenv("CASSANDRA_PASSWORD", "testyMcTesterson")
CASSANDRA_KEYSPACE = os.getenv("CASSANDRA_KEYSPACE", "vector_store")

# Configure authentication provider
auth_provider = PlainTextAuthProvider(username=CASSANDRA_USERNAME, password=CASSANDRA_PASSWORD)

# Connect to Cassandra with auth
cluster = Cluster(
    [CASSANDRA_HOST], 
    port=CASSANDRA_PORT,
    auth_provider=auth_provider
)
session = cluster.connect(CASSANDRA_KEYSPACE)


async def vector_search(query_text, top_k=5, content_type=None, repo_name=None):
    """Perform vector search in Cassandra based on query text with optional filters"""
    # Generate query vector
    qvec = embedding_model.encode(query_text, normalize_embeddings=True)
    qvec_list = qvec.tolist()

    # Base query
    base_cql = "SELECT content, metadata FROM vector_store.embeddings"
    where_clauses = []
    params = [qvec_list, top_k]

    # Add content_type filter if specified
    if content_type:
        where_clauses.append("metadata['content_type'] = %s")
        params.insert(1, content_type)

    # Add repo_name filter if specified
    if repo_name:
        where_clauses.append("metadata['repo'] = %s")
        params.insert(1, repo_name)

    # Build the complete query
    if where_clauses:
        cql = f"{base_cql} WHERE {' AND '.join(where_clauses)} ORDER BY embedding ANN OF %s LIMIT %s;"
    else:
        cql = f"{base_cql} ORDER BY embedding ANN OF %s LIMIT %s;"

    # Execute the query
    stmt = SimpleStatement(cql, fetch_size=top_k)
    rows = session.execute(stmt, params)

    # Format results
    results = []
    for r in rows:
        metadata_str = ""
        if hasattr(r, 'metadata') and r.metadata:
            metadata_str = "\nSource: " + ", ".join([f"{k}: {v}" for k, v in r.metadata.items()])
        results.append({"content": r.content, "metadata": r.metadata, "text": r.content + metadata_str})

    return results


async def generate_response(query, context):
    """Generate a response using the Qwen API with the given context"""
    prompt = INITIAL_PROMPT.format(context=context, query=query)

    # Call the Qwen API service
    response = requests.post(
        f"{QWEN_ENDPOINT}/v1/completions",
        headers={"Content-Type": "application/json"},
        json={
            "prompt": prompt,
            "max_tokens": 1024,  # Increased for more detailed reasoning
            "temperature": 0.7
        }
    )

    if response.status_code == 200:
        result = response.json()
        reply = result['choices'][0]['text'] if 'choices' in result else ""
        return {"answer": reply.strip()}
    else:
        return {"error": f"Error calling Qwen API: {response.status_code}", "details": response.text}


async def interactive_rag(query, initial_results, top_k=5):
    """Perform interactive RAG with follow-up queries"""
    # Format initial context
    context = "\n\n---\n\n".join(initial_results)

    # History of interactions for the conversation
    conversation_history = [
        {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT.format(query=query, context=context)}
    ]

    # Track search iterations
    iteration = 0
    max_iterations = MAX_ITERATIONS

    while iteration < max_iterations:
        # Call LLM with current conversation history
        response = requests.post(
            f"{QWEN_ENDPOINT}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "messages": conversation_history,
                "max_tokens": 1024,
                "temperature": 0.7
            }
        )

        if response.status_code != 200:
            return {"error": f"Error calling Qwen API: {response.status_code}", "details": response.text}

        # Get response content
        assistant_message = response.json()['choices'][0]['message']['content']
        conversation_history.append({"role": "assistant", "content": assistant_message})

        # Check for search request in the format [SEARCH: query]
        search_match = re.search(r'\[SEARCH:\s*(.+?)\]', assistant_message)

        if not search_match:
            # No more searches requested, return final answer
            return {"answer": assistant_message, "iterations": iteration}

        # Extract search query and perform vector search
        search_query = search_match.group(1).strip()
        follow_up_results = await vector_search(search_query, top_k)
        follow_up_context = "\n\n---\n\n".join(follow_up_results)

        # Add search results to conversation
        system_response = f"Search results for: {search_query}\n\n{follow_up_context}"
        conversation_history.append({"role": "system", "content": system_response})

        # Increment iteration counter
        iteration += 1

    # If we've reached max iterations, generate final response
    conversation_history.append({
        "role": "system", 
        "content": "You've reached the maximum number of search iterations. Please provide your final answer based on all information gathered so far."
    })

    # Get final response
    final_response = requests.post(
        f"{QWEN_ENDPOINT}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "messages": conversation_history,
            "max_tokens": 1024,
            "temperature": 0.7
        }
    )

    if final_response.status_code == 200:
        final_message = final_response.json()['choices'][0]['message']['content']
        return {"answer": final_message, "iterations": iteration, "max_iterations_reached": True}
    else:
        return {"error": f"Error in final response: {final_response.status_code}", "details": final_response.text}


@app.post("/rag")
async def rag(query: str, top_k: int = 5, filter_tags: list = None, enable_followup: bool = True, repo_name: str = None):
    """RAG endpoint that supports both simple and interactive approaches"""
    # Check if query is about a specific repository or asking for high-level info
    repo_specific = repo_name is not None or any(keyword in query.lower() for keyword in [
        "tell me about", "describe", "what is", "overview of", "summary of", "explain"
    ])

    # Initial context retrieval strategy based on query type
    initial_results = []

    if repo_specific:
        # If it's a high-level query about repos, prioritize repository summaries
        summary_results = await vector_search(
            query, 
            min(3, top_k),  # Get top repository summaries
            content_type="repository_summary",
            repo_name=repo_name
        )

        # Get directory summaries if needed
        dir_results = await vector_search(
            query,
            min(3, top_k),
            content_type="directory_summary",
            repo_name=repo_name
        )

        # Add code chunks for more specific details
        code_results = await vector_search(
            query,
            top_k - len(summary_results) - len(dir_results),
            content_type="code_chunk",
            repo_name=repo_name
        )

        # Combine results with summaries first
        initial_results = summary_results + dir_results + code_results
    else:
        # For specific technical questions, focus on code chunks
        initial_results = await vector_search(query, top_k)

    # Format results for context
    context_parts = [result["text"] for result in initial_results]
    context = "\n\n---\n\n".join(context_parts)

    if not enable_followup:
        # Use simple non-interactive approach
        return await generate_response(query, context)

    # Interactive RAG with follow-up queries
    return await interactive_rag(query, context_parts, top_k)
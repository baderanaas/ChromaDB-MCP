import os
import uuid
import hashlib
import logging
import logging.handlers
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, List, Optional, Any
import uvicorn

from mcp.server.fastmcp import FastMCP
from mcp.server import Server
from mcp.server.sse import SseServerTransport

from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.requests import Request

import chromadb
import chromadb.utils.embedding_functions as embedding_functions

load_dotenv()

PORT = int(os.getenv("PORT", "8000"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "chromadb_server.log")
LOG_MAX_SIZE = int(os.getenv("LOG_MAX_SIZE", "10485760"))
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")


# Configure logging
def setup_logging():
    """Setup comprehensive logging configuration"""
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, LOG_LEVEL))

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, LOG_LEVEL))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=LOG_MAX_SIZE, backupCount=LOG_BACKUP_COUNT
    )
    file_handler.setLevel(getattr(logging, LOG_LEVEL))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Set specific logger levels
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return logger


# Initialize logging
logger = setup_logging()

mcp = FastMCP(name="chromadb_server", host="0.0.0.0", port=PORT)

# Global variables for client and collections
client = None
collections = {}


def initialize_chromadb():
    """Initialize ChromaDB client and collections"""
    global client, collections

    logger.info("Initializing ChromaDB client...")

    try:
        persist_dir = os.getenv("CHROMADB_PERSIST_DIRECTORY", "chroma_db")
        logger.info(f"Using ChromaDB persist directory: {persist_dir}")

        client = chromadb.PersistentClient(path=persist_dir)
        logger.info("ChromaDB client initialized successfully")

        # Create embedding function
        embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY, model_name="text-embedding-3-small"
        )
        logger.info("OpenAI embedding function created")

        # Create or get collections
        collections = {
            "default_collection": _get_or_create_collection(
                "default_collection",
                "Default collection for storing memories",
                embedding_function,
            ),
            "user_collection": _get_or_create_collection(
                "user_collection",
                "Collection for user-specific memories",
                embedding_function,
            ),
        }

        logger.info(f"Collections initialized: {list(collections.keys())}")

    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {str(e)}", exc_info=True)
        raise


def _get_or_create_collection(
    name: str, description: Optional[str], embedding_function
):
    """Create or get a collection with proper metadata"""
    logger.debug(f"Getting or creating collection: {name}")

    try:
        metadata = {"hnsw:space": "cosine"}
        if description is not None:
            metadata["description"] = description

        collection = client.get_or_create_collection(
            name=name, metadata=metadata, embedding_function=embedding_function
        )

        logger.info(f"Collection '{name}' ready with {collection.count()} documents")
        return collection

    except Exception as e:
        logger.error(
            f"Failed to get/create collection '{name}': {str(e)}", exc_info=True
        )
        raise


def _get_collections():
    """Get list of available collections"""
    if client is None:
        logger.warning("Client not initialized when getting collections")
        return []

    try:
        collection_names = [collection.name for collection in client.list_collections()]
        logger.debug(f"Available collections: {collection_names}")
        return collection_names
    except Exception as e:
        logger.error(f"Failed to list collections: {str(e)}", exc_info=True)
        return []


@mcp.tool()
def store_memory(
    text: str,
    collection_name: str = "default_collection",
    metadata: Optional[Dict] = None,
) -> str:
    """Store a memory/text in the specified collection.

    This tool allows you to store text-based memories, notes, or information in a vector database
    for later retrieval using semantic search. Each memory is automatically embedded using OpenAI's
    text-embedding-3-small model.

    Args:
        text: The text content to store (required). Cannot be empty.
        collection_name: Name of the collection to store in. Options: "default_collection", "user_collection", or any custom collection created earlier. Defaults to "default_collection".
        metadata: Optional dictionary of additional metadata to attach to this memory (e.g., {"source": "user_input", "category": "important"}).

    Returns:
        Success message with the unique memory ID that can be used to delete this memory later.
    """
    logger.info(
        f"Storing memory in collection '{collection_name}' (length: {len(text)} chars)"
    )

    if client is None:
        logger.info("Client not initialized, auto-initializing...")
        initialize_chromadb()

    if not text.strip():
        logger.warning("Attempted to store empty text")
        raise ValueError("Cannot store empty text")

    if collection_name not in collections:
        logger.error(
            f"Unknown collection: {collection_name}. Available: {list(collections.keys())}"
        )
        raise ValueError(
            f"Unknown collection: {collection_name}. Available: {list(collections.keys())}"
        )

    try:
        memory_id = str(uuid.uuid4())
        logger.debug(f"Generated memory ID: {memory_id}")

        # Build metadata
        memory_metadata = {
            "timestamp": datetime.now().isoformat(),
            "content_hash": hashlib.md5(text.encode()).hexdigest(),
            "collection": collection_name,
            "character_count": len(text),
            "word_count": len(text.split()),
        }

        # Add any additional metadata provided, but filter out None values
        if metadata:
            for key, value in metadata.items():
                if value is not None:
                    memory_metadata[key] = value
            logger.debug(f"Added custom metadata: {metadata}")

        collection = collections[collection_name]
        collection.add(documents=[text], metadatas=[memory_metadata], ids=[memory_id])

        logger.info(f"Memory stored successfully with ID: {memory_id}")
        return f"Memory stored with ID: {memory_id}"

    except Exception as e:
        logger.error(f"Failed to store memory: {str(e)}", exc_info=True)
        raise


@mcp.tool()
def retrieve_memory(
    query: str,
    collection_name: str = "default_collection",
    limit: int = 5,
) -> Dict[str, Any]:
    """Retrieve memories from the specified collection using semantic search.

    This tool performs semantic search across stored memories to find the most relevant
    content based on the query. It uses vector similarity to find matches, so it can
    find semantically similar content even if the exact words don't match.

    Args:
        query: The search query text (required). Describe what you're looking for.
        collection_name: Name of the collection to search in. Options: "default_collection", "user_collection", or any custom collection. Defaults to "default_collection".
        limit: Maximum number of results to return (1-50). Defaults to 5.

    Returns:
        Dictionary containing:
        - query: The original search query
        - collection: The collection searched
        - results: ChromaDB results with documents, metadatas, distances, and ids
        - count: Number of results found
    """
    logger.info(
        f"Retrieving memories from '{collection_name}' with query: '{query}' (limit: {limit})"
    )

    if client is None:
        logger.info("Client not initialized, auto-initializing...")
        initialize_chromadb()

    if collection_name not in collections:
        logger.error(
            f"Unknown collection: {collection_name}. Available: {list(collections.keys())}"
        )
        raise ValueError(
            f"Unknown collection: {collection_name}. Available: {list(collections.keys())}"
        )

    # Validate limit
    if limit < 1 or limit > 50:
        logger.error(f"Invalid limit: {limit}. Must be between 1 and 50")
        raise ValueError("Limit must be between 1 and 50")

    try:
        collection = collections[collection_name]
        results = collection.query(query_texts=[query], n_results=limit)

        result_count = (
            len(results.get("documents", [[]])[0]) if results.get("documents") else 0
        )
        logger.info(f"Retrieved {result_count} memories for query")

        return {
            "query": query,
            "collection": collection_name,
            "results": results,
            "count": result_count,
        }

    except Exception as e:
        logger.error(f"Failed to retrieve memories: {str(e)}", exc_info=True)
        raise


@mcp.tool()
def list_collections() -> List[str]:
    """List all available collections in the ChromaDB database.

    This tool returns the names of all collections that exist in the database.
    Collections are used to organize different types of memories or information.

    Returns:
        List of collection names available for storing and retrieving memories.
    """
    logger.info("Listing all collections")

    if client is None:
        logger.info("Client not initialized, auto-initializing...")
        initialize_chromadb()

    collections_list = _get_collections()
    logger.info(f"Found {len(collections_list)} collections")
    return collections_list


@mcp.tool()
def create_collection(name: str, description: Optional[str] = None) -> str:
    """Create a new collection for organizing memories.

    This tool creates a new collection in the ChromaDB database. Collections are useful
    for organizing different types of information (e.g., user preferences, project notes,
    research data, etc.). Each collection maintains its own vector index.

    Args:
        name: Name of the new collection (required). Must be unique and contain only alphanumeric characters, underscores, and hyphens.
        description: Optional description of what this collection will be used for.

    Returns:
        Success message confirming collection creation or error if it already exists.
    """
    logger.info(f"Creating new collection: '{name}' with description: '{description}'")

    if client is None:
        logger.info("Client not initialized, auto-initializing...")
        initialize_chromadb()

    if name in collections:
        logger.warning(f"Collection '{name}' already exists")
        return f"Collection '{name}' already exists"

    # Validate collection name
    if not name.replace("_", "").replace("-", "").isalnum():
        logger.error(f"Invalid collection name: '{name}'")
        raise ValueError(
            "Collection name must contain only alphanumeric characters, underscores, and hyphens"
        )

    try:
        embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY, model_name="text-embedding-3-small"
        )

        collections[name] = _get_or_create_collection(
            name, description, embedding_function
        )
        logger.info(f"Collection '{name}' created successfully")
        return f"Collection '{name}' created successfully"

    except Exception as e:
        logger.error(f"Failed to create collection '{name}': {str(e)}", exc_info=True)
        raise


@mcp.tool()
def delete_collection(name: str) -> str:
    """Delete an entire collection and all its memories.

    This tool permanently deletes a collection and all the memories stored within it.
    Use with caution as this action cannot be undone. You cannot delete the default
    collections (default_collection, user_collection).

    Args:
        name: Name of the collection to delete (required).

    Returns:
        Success message confirming deletion or error message if collection doesn't exist.
    """
    global collections
    logger.warning(f"Attempting to delete collection: '{name}'")

    if client is None:
        logger.info("Client not initialized, auto-initializing...")
        initialize_chromadb()

    if name not in _get_collections():
        logger.error(f"Collection '{name}' does not exist")
        return f"Collection '{name}' does not exist"

    # Prevent deletion of default collections
    if name in ["default_collection", "user_collection"]:
        logger.warning(f"Attempted to delete default collection '{name}'")
        return f"Cannot delete default collection '{name}'"

    try:
        client.delete_collection(name)
        logger.info(f"Collection '{name}' deleted successfully")
        return f"Collection '{name}' deleted successfully"
    except Exception as e:
        logger.error(f"Error deleting collection '{name}': {str(e)}", exc_info=True)
        return f"Error deleting collection '{name}': {str(e)}"


@mcp.tool()
def delete_memory(memory_id: str, collection_name: str = "default_collection") -> str:
    """Delete a specific memory by its unique ID.

    This tool removes a single memory from the specified collection using its unique ID.
    The memory ID is returned when you store a memory using store_memory().

    Args:
        memory_id: The unique identifier of the memory to delete (required).
        collection_name: Name of the collection containing the memory. Defaults to "default_collection".

    Returns:
        Success message confirming deletion or error message if memory doesn't exist.
    """
    logger.info(f"Deleting memory '{memory_id}' from collection '{collection_name}'")

    if client is None:
        logger.info("Client not initialized, auto-initializing...")
        initialize_chromadb()

    if collection_name not in _get_collections():
        logger.error(
            f"Unknown collection: {collection_name}. Available: {list(collections.keys())}"
        )
        raise ValueError(
            f"Unknown collection: {collection_name}. Available: {list(collections.keys())}"
        )

    try:
        collection = collections[collection_name]
        collection.delete(ids=[memory_id])
        logger.info(f"Memory {memory_id} deleted successfully")
        return f"Memory {memory_id} deleted successfully"
    except Exception as e:
        logger.error(f"Error deleting memory {memory_id}: {str(e)}", exc_info=True)
        return f"Error deleting memory {memory_id}: {str(e)}"



## Using streamable-http transport for MCP server
# Initialize ChromaDB when the module is imported
# if __name__ == "__main__":
#     logger.info("Starting ChromaDB MCP Server...")
#     logger.info(f"Log level: {LOG_LEVEL}")
#     logger.info(f"Log file: {LOG_FILE}")
#     logger.info(f"Port: {PORT}")

#     try:
#         initialize_chromadb()
#         logger.info(f"ChromaDB MCP Server starting on port {PORT}")
#         logger.info(f"Available collections: {list(collections.keys())}")
#         mcp.run(transport="streamable-http")
#     except Exception as e:
#         logger.critical(f"Failed to start server: {str(e)}", exc_info=True)
#         raise


## Using sse transport for MCP server
def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """
    Constructs a Starlette app with SSE and message endpoints.

    Args:
        mcp_server (Server): The core MCP server instance.
        debug (bool): Enable debug mode for verbose logs.

    Returns:
        Starlette: The full Starlette app with routes.
    """
    # Create SSE transport handler to manage long-lived SSE connections
    sse = SseServerTransport("/messages/")

    # This function is triggered when a client connects to `/sse`
    async def handle_sse(request: Request) -> None:
        """
        Handles a new SSE client connection and links it to the MCP server.
        """
        # Open an SSE connection, then hand off read/write streams to MCP
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,  # Low-level send function provided by Starlette
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    # Return the Starlette app with configured endpoints
    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),  # For initiating SSE connection
            Mount(
                "/messages/", app=sse.handle_post_message
            ),  # For POST-based communication
        ],
    )


if __name__ == "__main__":
    mcp_server = mcp._mcp_server

    # Command-line arguments for host/port control
    import argparse

    parser = argparse.ArgumentParser(description="Run MCP SSE-based server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=PORT, help="Port to listen on")
    args = parser.parse_args()

    # Build the Starlette app with debug mode enabled
    starlette_app = create_starlette_app(mcp_server, debug=True)

    # Launch the server using Uvicorn
    uvicorn.run(starlette_app, host=args.host, port=args.port)

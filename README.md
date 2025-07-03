# ChromaDB-MCP

A fast, extensible server for storing and retrieving semantic memories using **ChromaDB** and **OpenAI embeddings**, exposed via the **Model Context Protocol (MCP)** with support for SSE and streamable HTTP transports.

---

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Requirements](#requirements)
* [Installation](#installation)
* [Configuration](#configuration)
* [Usage](#usage)
* [Available MCP Tools](#available-mcp-tools)
* [Architecture](#architecture)
* [Development](#development)

---

## Overview

This project implements an MCP server that uses **ChromaDB** as a persistent vector database backend to store and retrieve text memories embedded via OpenAI's `text-embedding-3-small` model.
It supports multiple collections for organizing memories and exposes rich tools for storing, retrieving, listing, creating, and deleting memories and collections.

You can try the MCP server live and hosted at:
https://chromadb-mcp.onrender.com

The server supports two transports:

* **Server-Sent Events (SSE)** (default in the example)
* **Streamable HTTP** (commented-out example provided)

The server is built with:

* Python 3.9+
* [FastMCP](https://github.com/theailanguage/mcp) for MCP protocol
* [Starlette](https://www.starlette.io/) and [uvicorn](https://www.uvicorn.org/) for async HTTP serving
* [ChromaDB](https://chromadb.com/) as vector store
* OpenAI API for embeddings

---

## Usage

You can try the hosted MCP server without any setup at:
**[https://chromadb-mcp.onrender.com](https://chromadb-mcp.onrender.com)**

It exposes MCP via:

* `GET /sse` — Server-Sent Events endpoint
* `POST /messages/` — POST message endpoint

For local usage, run the server yourself:

```bash
python server.py --host 0.0.0.0 --port 8000
```

This will start the MCP server locally with SSE transport enabled.

---

## Client Usage

You can use the MCP client with this server via Python or **Gemini's `uv` CLI**.

### Using Python

```bash
python client.py https://chromadb-mcp.onrender.com
```

### Using Gemini's `uv` CLI

If you have [Gemini](https://github.com/tern-tools/gemini) and `uv` installed:

```bash
uv run client.py https://chromadb-mcp.onrender.com
```

This will connect the client to the hosted MCP server and allow you to interact with it using the defined tools (e.g., `store_memory`, `retrieve_memory`, etc.).

---

## Features

* Store textual memories with semantic embeddings
* Retrieve memories via semantic search queries
* Organize memories into named collections
* Create and delete collections (except protected defaults)
* Delete individual memories by ID
* Detailed, configurable logging
* SSE-based streaming MCP server endpoint
* Streamable HTTP transport support (optional)

---

## Requirements

* Python 3.9 or higher
* Access to OpenAI API (API key required)
* [Chromadb Python package](https://pypi.org/project/chromadb/)
* [FastMCP package](https://github.com/theailanguage/mcp)
* Uvicorn for ASGI serving

---

## Installation

1. Clone this repo:

   ```bash
   git clone https://github.com/baderanaas/ChromaDB-MCP.git
   cd ChromaDB-MCP
   ```

2. Create and activate a Python virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root with:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   PORT=8000
   LOG_LEVEL=INFO
   LOG_FILE=chromadb_server.log
   LOG_MAX_SIZE=10485760
   LOG_BACKUP_COUNT=5
   CHROMADB_PERSIST_DIRECTORY=chroma_db
   GEMINI_API_KEY=your_gemini_api_key
   ```

---

## Configuration

* **OPENAI\_API\_KEY**: Your OpenAI API key for embedding generation
* **PORT**: Port to run the server on (default 8000)
* **LOG\_LEVEL**: Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
* **LOG\_FILE**: Path to log file
* **LOG\_MAX\_SIZE**: Maximum size (bytes) for log rotation
* **LOG\_BACKUP\_COUNT**: Number of rotated log backups to keep
* **CHROMADB\_PERSIST\_DIRECTORY**: Directory for persistent ChromaDB data

---

## Usage

Run the MCP SSE server:

```bash
python server.py --host 0.0.0.0 --port 8000
```

This will start the MCP server with SSE transport exposed at:

* `GET /sse` - Server-Sent Events endpoint for MCP
* `POST /messages/` - MCP message endpoint for sending requests

You can connect MCP clients to these endpoints for communication.

---

## Available MCP Tools

### store\_memory

Store a memory (text) in a specified collection.

**Arguments:**

* `text` (str): Text content to store
* `collection_name` (str, default "default\_collection"): Collection to store in
* `metadata` (dict, optional): Additional metadata

Returns: Memory ID string

---

### retrieve\_memory

Retrieve memories relevant to a query via semantic search.

**Arguments:**

* `query` (str): Search query text
* `collection_name` (str, default "default\_collection"): Collection to search
* `limit` (int, default 5): Max results to return (1-50)

Returns: Dict with documents, metadatas, distances, ids

---

### list\_collections

List all available collections.

Returns: List of collection names

---

### create\_collection

Create a new named collection.

**Arguments:**

* `name` (str): Collection name (alphanumeric, underscore, hyphen)
* `description` (str, optional): Description

Returns: Success message or error

---

### delete\_collection

Delete a named collection (except default ones).

**Arguments:**

* `name` (str): Collection name

Returns: Success or error message

---

### delete\_memory

Delete a memory by its ID in a collection.

**Arguments:**

* `memory_id` (str): Memory UUID
* `collection_name` (str, default "default\_collection")

Returns: Success or error message

---

## Architecture

* **FastMCP** manages MCP protocol messages
* **ChromaDB** handles vector database persistence and search
* **OpenAI Embeddings** embed text to vectors for semantic indexing
* **Starlette + Uvicorn** serve HTTP and SSE endpoints
* **Logging** with rotating file and console outputs

## Development

* Ensure `.env` is configured with API keys
* Use `uvicorn server:starlette_app --reload` for live reload during development
* Extend MCP tools by adding new functions decorated with `@mcp.tool()`
* Use logging for debug and info traces

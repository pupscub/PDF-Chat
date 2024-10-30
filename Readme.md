# RAG-Powered PDF Chat

An offline PDF chat application built using LangChain and OLLAMA that enables natural conversations with multiple PDF documents using Retrieval Augmented Generation (RAG).

## Features

- **Multi-PDF Support**: Chat with multiple PDFs simultaneously
- **Offline Operation**: Runs completely offline using OLLAMA
- **Advanced RAG Pipeline**: Leverages state-of-the-art embedding and retrieval techniques
- **Context-Aware Responses**: Maintains conversation context for more relevant answers
- **Document Management**: Easy PDF upload and management interface

## Architecture

**Core Components**:
- LangChain for RAG pipeline orchestration
- OLLAMA for local LLM inference
- Document loaders and text splitters
- Vector store for efficient retrieval
- Custom prompt templates

## Installation

```bash
# Clone the repository
git clone https://github.com/username/rag-pdf-chat
cd rag-pdf-chat

# Install dependencies
pip install -r requirements.txt

# Install OLLAMA (if not already installed)
curl https://ollama.ai/install.sh | sh

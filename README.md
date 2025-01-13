# rag-agent
Basic Retrieval Augmented Generation agent in Python.

# Install
curl -fsSL https://ollama.com/install.sh | sh

# Start service
ollama serve

If Ollama is already running. You can verify with:

curl http://localhost:11434/api/tags

If needed, restart Ollama:

sudo systemctl restart ollama

# In new terminal, pull models you want to use
ollama pull llama2
ollama pull mistral
# etc for other models

The Ollama model "nomic-embed-text" that's being used for embeddings must be available on your system. You need to pull (download) this model first using Ollama.
Here's how to do it:

First, make sure Ollama is running on your system
Then, open a terminal and run:

ollama pull nomic-embed-text

This will download the nomic-embed-text model that's needed for generating embeddings.

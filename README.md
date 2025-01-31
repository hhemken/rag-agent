# RAG Interface

A Flask-based web application for Retrieval Augmented Generation (RAG) that allows users to manage document databases, interact with different language models, and perform document queries through a user-friendly interface.

## Features

- **Document Database Management**
  - Upload and process PDF documents
  - Create and manage multiple vector databases using Chroma
  - Reset or update existing databases
  - Configurable data and storage paths

- **Chat Interface**
  - Interactive question-answering system
  - Support for multiple language models via Ollama
  - Dynamic model selection
  - Source tracking and citation
  - Real-time response generation

- **Vector Search**
  - Semantic search capabilities using embeddings
  - Configurable chunk size and overlap for document splitting
  - Support for multiple embedding models

## Prerequisites

- Python 3.8+
- Flask
- Ollama (for local LLM support)
- Chrome DB
- Required Python packages (see Installation section)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-interface
```

2. Install the required Python packages:
```bash
pip install flask langchain-community langchain-text-splitters langchain-ollama chromadb
```

3. Install and start Ollama (if not already installed):
```bash
# Follow instructions at https://ollama.ai/download
```

4. Configure the application:
   - Set your desired paths in `config` dictionary in `app.py`
   - Ensure Ollama is running and accessible

## Project Structure

```
.
├── app.py                  # Main Flask application
├── get_embedding_function.py   # Embedding model configuration
├── populate_database.py    # Database management utilities
├── query_data.py          # RAG query processing
├── test_rag.py            # Test suite
└── templates/             # HTML templates
    ├── base.html         
    ├── chat.html
    ├── database.html
    └── index.html
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Access the web interface at `http://localhost:5000`

3. Database Management:
   - Navigate to the Database Management page
   - Set your Chroma and Data paths
   - Upload PDF documents to your data path
   - Process the database (with optional reset)

4. Chat Interface:
   - Select your desired database and language model
   - Enter your question
   - View responses with source citations

## Configuration

Key configuration options in `app.py`:

```python
config = {
    "CHROMA_PATH": "chroma",
    "DATA_PATH": "/path/to/docs",
    "LLM_TO_USE": "llama2:13b"
}
```

- `CHROMA_PATH`: Directory for vector database storage
- `DATA_PATH`: Directory containing PDF documents
- `LLM_TO_USE`: Default language model to use

## Customization

### Embedding Models

Modify `get_embedding_function.py` to use different embedding models:

```python
embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

### Document Processing

Adjust chunk size and overlap in `populate_database.py`:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=80,
    length_function=len,
    is_separator_regex=False,
)
```

## Testing

Run the test suite:
```bash
python test_rag.py
```

## Troubleshooting

1. **Database Connection Issues**
   - Ensure Chroma path is correctly set
   - Check write permissions for the database directory

2. **Model Loading Errors**
   - Verify Ollama is running
   - Check model availability using `/api/list_models`

3. **PDF Processing Issues**
   - Ensure PDFs are readable and not corrupted
   - Check DATA_PATH permissions

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

[Insert your chosen license here]

## Acknowledgments

- LangChain for the RAG implementation framework
- Ollama for local LLM support
- ChromaDB for vector storage
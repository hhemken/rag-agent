import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma


CHROMA_PATH = "chroma"
# DATA_PATH = "/home/hemkenhg/workspace/books/rpi"
DATA_PATH = "/home/hemkenhg/workspace/books/woodworking"


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    documents = []
    for document in os.listdir(DATA_PATH):
        document_path = os.path.abspath(os.path.join(DATA_PATH, document))
        if document.endswith(".pdf"):
            print(f'loading: {document_path}')
            loader = PyPDFLoader(document_path)
            # Extend the list instead of appending
            documents.extend(loader.load())
    return documents


def split_documents(documents: list[Document], chunking_method='recursive', **kwargs):
    """
    Split documents using specified chunking method.

    Args:
        documents: List of documents to split
        chunking_method: 'recursive' or 'semantic'
        **kwargs: Additional arguments for the chunker
    """
    if chunking_method == 'semantic':
        # Filter only semantic chunking parameters
        semantic_params = {
            'n_clusters': kwargs.get('n_clusters'),
            'min_chunk_size': kwargs.get('min_chunk_size', 100),
            'max_chunk_size': kwargs.get('max_chunk_size', 1000)
        }
        # Remove None values
        semantic_params = {k: v for k, v in semantic_params.items() if v is not None}

        from semantic_chunking import SemanticChunker
        chunker = SemanticChunker(**semantic_params)
        return chunker.split_documents(documents)
    else:
        # Filter only recursive chunking parameters
        recursive_params = {
            'chunk_size': kwargs.get('chunk_size', 800),
            'chunk_overlap': kwargs.get('chunk_overlap', 80)
        }

        text_splitter = RecursiveCharacterTextSplitter(
            **recursive_params,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        # db.persist()
    else:
        print("✅ No new documents to add")

def add_documents_to_chroma(chunks, db):
    """Modified version of add_to_chroma that takes db instance as parameter"""
    # Calculate chunk IDs
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Get existing items from the database
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Add documents that don't exist in the DB
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if new_chunks:
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()

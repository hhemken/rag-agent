from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import ConvergenceWarning
from langchain.schema.document import Document
import numpy as np
from typing import List
import re
import warnings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
Enhanced PDF processing with detailed debugging
"""

class SemanticChunker:
    def __init__(self, n_clusters=None, min_chunk_size=100, max_chunk_size=2000):
        self.n_clusters = n_clusters
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            min_df=1,
            max_df=0.95,
            token_pattern=r'(?u)\b\w+\b',
            strip_accents='unicode'
        )

        # Set up detailed logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _clean_pdf_text(self, text: str) -> str:
        """Enhanced PDF text cleaning with logging"""
        if not text:
            self.logger.warning("Received empty text for cleaning")
            return ""

        self.logger.info(f"Original text length: {len(text)}")
        self.logger.debug(f"First 500 chars of original text: {text[:500]}")

        # Initial cleaning
        text = text.replace('\f', ' ')
        text = text.replace('\x0c', ' ')

        # Handle PDF-specific artifacts
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # Fix hyphenation
        text = re.sub(r'\s*\n\s*', ' ', text)  # Normalize newlines to spaces
        text = re.sub(r'\s+', ' ', text)  # Normalize multiple spaces

        # Remove header/footer artifacts (common in PDFs)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # Page numbers
        text = re.sub(r'^\s*[A-Za-z0-9\s\-_]+\s*$', '', text, flags=re.MULTILINE)  # Headers/footers

        # Clean text
        cleaned_text = text.strip()
        self.logger.info(f"Cleaned text length: {len(cleaned_text)}")
        self.logger.debug(f"First 500 chars of cleaned text: {cleaned_text[:500]}")

        return cleaned_text

    def _split_into_chunks(self, text: str) -> List[str]:
        """Enhanced text splitting with debugging"""
        if not text:
            self.logger.warning("Received empty text for splitting")
            return []

        self.logger.info(f"Starting text splitting. Text length: {len(text)}")

        # Clean the text first
        text = self._clean_pdf_text(text)
        if not text.strip():
            self.logger.warning("Text is empty after cleaning")
            return []

        # Split on major section boundaries first
        chunks = []

        # Try to split on chapter markers first
        chapter_splits = re.split(r'(?i)(chapter\s+[0-9]+|section\s+[0-9]+)', text)
        if len(chapter_splits) > 1:
            self.logger.info("Found chapter/section markers")
            # Reassemble splits with their headers
            for i in range(1, len(chapter_splits), 2):
                if i+1 < len(chapter_splits):
                    chunk = chapter_splits[i] + chapter_splits[i+1]
                    if chunk.strip():
                        chunks.append(chunk.strip())
        else:
            self.logger.info("No chapter markers found, trying paragraph splitting")
            # Split on double newlines (paragraphs)
            paragraphs = re.split(r'\n\s*\n', text)
            current_chunk = []
            current_length = 0

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                if current_length + len(para) <= self.max_chunk_size:
                    current_chunk.append(para)
                    current_length += len(para)
                else:
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        if len(chunk_text) >= self.min_chunk_size:
                            chunks.append(chunk_text)
                    current_chunk = [para]
                    current_length = len(para)

            # Add the last chunk if it exists
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(chunk_text)

        self.logger.info(f"Created {len(chunks)} initial chunks")
        if not chunks:
            self.logger.warning("No chunks created after splitting")
            if len(text) >= self.min_chunk_size:
                self.logger.info("Using entire text as one chunk")
                chunks = [text]

        # Log chunk statistics
        for i, chunk in enumerate(chunks):
            self.logger.info(f"Chunk {i}: {len(chunk)} characters")
            self.logger.debug(f"Chunk {i} preview: {chunk[:200]}...")

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Enhanced document splitting with detailed logging"""
        all_chunks = []
        total_docs = len(documents)

        self.logger.info(f"Processing {total_docs} documents")

        for idx, doc in enumerate(documents, 1):
            try:
                self.logger.info(f"Processing document {idx}/{total_docs}")
                self.logger.info(f"Document metadata: {doc.metadata}")

                if not doc.page_content:
                    self.logger.warning(f"Document {idx} has no content")
                    continue

                self.logger.info(f"Document {idx} content length: {len(doc.page_content)}")

                # Get initial chunks
                chunks = self._split_into_chunks(doc.page_content)

                if not chunks:
                    self.logger.warning(f"No chunks created for document {idx}")
                    # Store entire document as one chunk if it's not too small
                    if len(doc.page_content.strip()) >= self.min_chunk_size:
                        chunks = [self._clean_pdf_text(doc.page_content)]
                        self.logger.info("Using entire document as one chunk")

                for chunk_idx, chunk in enumerate(chunks):
                    chunk_doc = Document(
                        page_content=chunk,
                        metadata={
                            **doc.metadata,
                            'chunk_type': 'semantic',
                            'chunk_index': chunk_idx,
                            'total_chunks': len(chunks),
                            'original_length': len(doc.page_content),
                            'chunk_length': len(chunk)
                        }
                    )
                    all_chunks.append(chunk_doc)

                self.logger.info(f"Created {len(chunks)} chunks for document {idx}")

            except Exception as e:
                self.logger.error(f"Error processing document {idx}: {str(e)}")
                # Store entire document as one chunk in case of error
                chunk_doc = Document(
                    page_content=self._clean_pdf_text(doc.page_content),
                    metadata={
                        **doc.metadata,
                        'chunk_type': 'error_fallback',
                        'error': str(e)
                    }
                )
                all_chunks.append(chunk_doc)

        self.logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks

def debug_pdf_loading(file_path: str):
    """
    Debug function to test PDF loading
    Add this to your app.py or where you load PDFs
    """
    from langchain_community.document_loaders import PyPDFLoader
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Attempting to load PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        logger.info(f"Successfully loaded {len(pages)} pages")
        for i, page in enumerate(pages):
            logger.info(f"Page {i+1} length: {len(page.page_content)}")
            logger.debug(f"Page {i+1} preview: {page.page_content[:200]}...")

        return pages
    except Exception as e:
        logger.error(f"Error loading PDF: {str(e)}")
        return None

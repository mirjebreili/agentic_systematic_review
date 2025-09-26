"""
Handle PDF text extraction and intelligent chunking using LangChain.
"""

from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Optional
from langchain_core.documents import Document
import logging
import hashlib

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def extract_text_from_pdf(self, pdf_path: str) -> List[Document]:
        """
        Extract text and metadata from a PDF file using PDFPlumber.
        """
        logger.info(f"Extracting text from {pdf_path}")
        try:
            loader = PDFPlumberLoader(pdf_path)
            documents = loader.load()
            logger.info(f"Extracted {len(documents)} pages from {pdf_path}")
            return documents
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            raise

    def create_smart_chunks(self, documents: List[Document], paper_id: str) -> List[Document]:
        """
        Create chunks from a list of documents with rich metadata.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        chunks = self.text_splitter.create_documents(texts, metadatas=metadatas)

        # Add custom metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata["paper_id"] = paper_id
            chunk.metadata["chunk_index"] = i

            # Create a unique ID for the chunk
            content_to_hash = f"{paper_id}-{chunk.metadata['page']}-{i}-{chunk.page_content}"
            chunk.metadata["chunk_id"] = hashlib.sha256(content_to_hash.encode('utf-8')).hexdigest()

        logger.info(f"Created {len(chunks)} chunks for paper {paper_id}")
        return chunks

    def extract_and_chunk_pdf(self, pdf_path: str, paper_id: str) -> Optional[List[Document]]:
        """Extract a PDF and generate enriched chunks, returning ``None`` on failure."""

        try:
            documents = self.extract_text_from_pdf(pdf_path)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "Error extracting text from %s: %s", pdf_path, exc, exc_info=True
            )
            return None

        try:
            return self.create_smart_chunks(documents, paper_id)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "Error chunking extracted text for %s: %s", pdf_path, exc, exc_info=True
            )
            return None

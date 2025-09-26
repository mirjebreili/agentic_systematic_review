"""
Manage ChromaDB collections and embeddings for each paper.
"""

try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List, Optional
from langchain_core.documents import Document
from pathlib import Path
import logging
import chromadb

from pdf_processor import PDFProcessor

logger = logging.getLogger(__name__)

class EmbeddingManager:
    def __init__(self, chroma_db_path: Path, embedding_model_name: str, pdf_processor: PDFProcessor):
        """Initialize embedding model and ChromaDB client."""
        self.chroma_db_path = str(chroma_db_path.resolve())
        self.embedding_model_name = embedding_model_name
        self.pdf_processor = pdf_processor

        chroma_db_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing Ollama embedding model: {embedding_model_name}")

        try:
            self.embedding_function = OllamaEmbeddings(
                base_url="http://localhost:11434",  # Default Ollama URL
                model=self.embedding_model_name,
                show_progress=True,
            )

            # Test the connection but don't crash if it fails
            try:
                test_embedding = self.embedding_function.embed_query("test")
                logger.info(f"Ollama embedding model loaded successfully! Embedding dimension: {len(test_embedding)}")
            except Exception as e:
                logger.warning(f"Could not connect to Ollama to test embedding model: {e}")

        except Exception as e:
            logger.error(f"Failed to initialize Ollama embeddings class: {e}")
            raise

        self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
        logger.info(f"ChromaDB client initialized at path: {self.chroma_db_path}")

    def get_or_create_collection(
        self,
        paper_id: str,
        pdf_path: str,
        chunks: Optional[List[Document]] = None,
    ) -> Optional[Chroma]:
        """Check if collection exists, create if not"""
        try:
            # Try to get existing collection first
            existing_collection = self.chroma_client.get_collection(name=paper_id)
            logger.info(f"Found existing collection for paper: {paper_id}")

            # Return ChromaDB wrapper for existing collection
            return Chroma(
                client=self.chroma_client,
                collection_name=paper_id,
                embedding_function=self.embedding_function
            )

        except Exception as e:
            # Collection doesn't exist, create it
            logger.info(f"Collection not found for {paper_id}, creating new one...")
            return self.create_paper_collection(paper_id, pdf_path, chunks=chunks)

    def create_paper_collection(
        self,
        paper_id: str,
        pdf_path: str,
        chunks: Optional[List[Document]] = None,
    ) -> Optional[Chroma]:
        """Create new collection and embed chunks"""
        try:
            logger.info(f"Preparing collection for paper: {paper_id}")

            if chunks is None:
                logger.debug("No precomputed chunks provided for %s. Extracting afresh.", paper_id)
                chunks = self.pdf_processor.extract_and_chunk_pdf(pdf_path, paper_id)

            if not chunks:
                logger.error(
                    "No chunks available for %s. Skipping collection creation.", paper_id
                )
                return None

            for chunk in chunks:
                chunk.metadata.setdefault("source", pdf_path)
                if "page_number" not in chunk.metadata and "page" in chunk.metadata:
                    chunk.metadata["page_number"] = chunk.metadata["page"]

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_function,
                client=self.chroma_client,
                collection_name=paper_id,
            )

            logger.info(f"Created collection '%s' with %d chunks", paper_id, len(chunks))
            return vectorstore

        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "Failed to create collection for %s from %s: %s", paper_id, pdf_path, exc, exc_info=True
            )
            return None

    def semantic_search(self, collection: Chroma, query: str, k: int = 5) -> List[Document]:
        """Perform semantic search and return relevant chunks."""
        if not collection:
            logger.warning("Cannot perform search on a null collection.")
            return []

        logger.debug(f"Performing semantic search for query: '{query}' with k={k}")
        try:
            relevant_chunks = collection.similarity_search(query, k=k)
            logger.info(f"Found {len(relevant_chunks)} relevant chunks for query: '{query}'")
            return relevant_chunks
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def delete_collection(self, paper_id: str):
        """Delete a collection from ChromaDB."""
        try:
            self.chroma_client.delete_collection(name=paper_id)
            logger.info(f"Successfully deleted collection: {paper_id}")
        except ValueError:
            logger.warning(f"Attempted to delete non-existent collection: {paper_id}")
        except Exception as e:
            logger.error(f"Failed to delete collection {paper_id}: {e}")
            raise

    def list_collections(self) -> List[str]:
        """List all available collections in ChromaDB."""
        return [col.name for col in self.chroma_client.list_collections()]

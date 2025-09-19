"""
Manage ChromaDB collections and embeddings for each paper.
"""

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fallback for older versions
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List
from langchain_core.documents import Document
from pathlib import Path
import logging
import chromadb

from pdf_processor import PDFProcessor

logger = logging.getLogger(__name__)

class EmbeddingManager:
    def __init__(self, chroma_db_path: str, embedding_model: str, pdf_processor: PDFProcessor):
        """Initialize embedding model and ChromaDB client."""
        self.chroma_db_path = str(Path(chroma_db_path).resolve())
        self.embedding_model_name = embedding_model
        self.pdf_processor = pdf_processor

        logger.info(f"Initializing embedding model: {self.embedding_model_name}")
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cpu'}, # Explicitly use CPU
            encode_kwargs={'normalize_embeddings': True}
        )

        self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
        logger.info(f"ChromaDB client initialized at path: {self.chroma_db_path}")

    def get_or_create_collection(self, paper_id: str, pdf_path: str) -> Chroma:
        """
        Get an existing collection or create a new one if it doesn't exist.
        This handles caching by not reprocessing existing papers.
        """
        try:
            # Check if the collection already exists
            self.chroma_client.get_collection(name=paper_id)
            logger.info(f"Collection '{paper_id}' already exists. Loading from disk.")
            collection = Chroma(
                client=self.chroma_client,
                collection_name=paper_id,
                embedding_function=self.embedding_function,
            )
            return collection
        except ValueError:
            # Collection does not exist, so create it
            logger.info(f"Collection '{paper_id}' not found. Creating new collection.")

            # Process the PDF to get chunks
            documents = self.pdf_processor.extract_text_from_pdf(pdf_path)
            chunks = self.pdf_processor.create_smart_chunks(documents, paper_id)

            if not chunks:
                logger.warning(f"No chunks were created for {pdf_path}. Cannot create collection.")
                return None

            # Create a new ChromaDB collection
            collection = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_function,
                collection_name=paper_id,
                persist_directory=self.chroma_db_path,
            )
            logger.info(f"Successfully created and persisted collection '{paper_id}' with {len(chunks)} chunks.")
            return collection

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

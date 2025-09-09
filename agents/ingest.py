"""
The Ingest Agent is responsible for reading documents from file, parsing their
content, and breaking them down into processable chunks.
"""
import logging
from pathlib import Path
from typing import Dict

from src.io.parsers import parse_document
from src.io.chunking import chunk_page
from src.schemas.output_schemas import AppState

def ingest_documents(state: Dict) -> Dict:
    """
    The entry point for the Ingestion agent.

    This agent performs the following steps:
    1. Iterates through the list of documents specified in `state.documents_to_process`.
    2. Uses the `parsers` module to extract text content and page information.
    3. Uses the `chunking` module to split the extracted text into smaller chunks.
    4. Populates the `state.document_chunks` and `state.processing_errors` fields.
    5. Returns the updated state.

    Args:
        state: The current application state dictionary.

    Returns:
        The updated application state dictionary.
    """
    app_state = AppState(**state)
    logging.info("--- Starting Ingestion Stage ---")

    for doc_path_str in app_state.documents_to_process:
        doc_path = Path(doc_path_str)
        doc_id = doc_path.name
        logging.info(f"Processing: {doc_id}")

        try:
            pages = parse_document(doc_path)
            if not pages:
                logging.warning(f"Document '{doc_id}' could not be parsed or is empty.")
                app_state.processing_errors[doc_id] = "Document is empty or could not be parsed."
                continue

            doc_chunks = []
            for page in pages:
                doc_chunks.extend(chunk_page(doc_id=doc_id, page=page))

            if not doc_chunks:
                logging.warning(f"No content chunks extracted from document: {doc_id}")
                app_state.processing_errors[doc_id] = "No content could be extracted into chunks."
                continue

            app_state.document_chunks[doc_id] = doc_chunks
            logging.info(f"Successfully ingested and chunked '{doc_id}' into {len(doc_chunks)} chunks.")

        except Exception as e:
            logging.error(f"Failed to ingest or chunk document '{doc_id}': {e}", exc_info=True)
            app_state.processing_errors[doc_id] = f"An unexpected error occurred: {e}"

    logging.info(f"--- Ingestion Stage Complete. Processed {len(app_state.documents_to_process)} documents. ---")
    return app_state.dict()

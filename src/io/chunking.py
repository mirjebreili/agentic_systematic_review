"""
Handles the logic for splitting parsed document pages into smaller,
manageable chunks for the LLM to process.
"""
import logging
from typing import List

from src.io.parsers import Page
from src.schemas.output_schemas import Chunk

def chunk_page(document_id: str, page: Page) -> List[Chunk]:
    """
    Splits the text of a single Page object into smaller Chunk objects.

    This implementation uses a simple paragraph-based chunking strategy.
    Future improvements could involve more sophisticated methods like
    fixed-size overlapping chunks or recursive chunking.

    Args:
        document_id: The identifier of the document the page belongs to.
        page: The Page object to be chunked.

    Returns:
        A list of Chunk objects.
    """
    chunks: List[Chunk] = []
    # Split by double newline, which typically separates paragraphs.
    paragraphs = page.text.split('\n\n')

    for i, para_text in enumerate(paragraphs):
        stripped_text = para_text.strip()
        if stripped_text:  # Ignore empty paragraphs
            chunk_id = f"{document_id}_p{page.page_number}_c{i}"
            chunks.append(Chunk(
                document_id=document_id,
                chunk_id=chunk_id,
                page_number=page.page_number,
                text=stripped_text
            ))

    if not chunks:
        logging.warning(f"Page {page.page_number} of document '{document_id}' resulted in zero chunks.")

    return chunks

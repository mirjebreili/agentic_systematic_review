"""
Handles parsing of different document formats (.pdf, .docx, .html, .txt)
into a unified text format for further processing. Includes OCR capabilities
for scanned PDFs.
"""
import logging
from pathlib import Path
from typing import List, Dict, Callable

import fitz  # PyMuPDF
import docx
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract

from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Data Structure for Parsed Content ---

class Page(BaseModel):
    """Represents the content of a single page from a document."""
    page_number: int = Field(..., description="The page number in the document (1-indexed).")
    text: str = Field(..., description="The full text content of the page.")
    source: str = Field(..., description="The method used to extract the text (e.g., 'text', 'ocr').")

# --- Individual Parser Implementations ---

def _parse_pdf(file_path: Path) -> List[Page]:
    """
    Parses a PDF document, attempting to extract text directly and falling
    back to OCR if a page appears to be scanned.
    """
    pages: List[Page] = []
    try:
        doc = fitz.open(file_path)
        for i, page_obj in enumerate(doc):
            page_num = i + 1
            text = page_obj.get_text()
            source = "text"

            # If text is minimal, assume it's a scanned image and try OCR
            if len(text.strip()) < 100:
                logging.info(f"Page {page_num} of '{file_path.name}' has little text; attempting OCR.")
                try:
                    pix = page_obj.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = pytesseract.image_to_string(img)
                    if len(ocr_text.strip()) > len(text.strip()):
                        text = ocr_text
                        source = "ocr"
                        logging.info(f"OCR successful for page {page_num} of '{file_path.name}'.")
                except Exception as e:
                    logging.error(f"OCR failed for page {page_num} of '{file_path.name}': {e}", exc_info=True)

            pages.append(Page(page_number=page_num, text=text, source=source))
        doc.close()
    except Exception as e:
        logging.error(f"Failed to parse PDF '{file_path.name}': {e}", exc_info=True)
        # Return an empty list or re-raise depending on desired error handling
    return pages

def _parse_docx(file_path: Path) -> List[Page]:
    """Parses a .docx document. Lacks pagination, so returns as a single page."""
    try:
        document = docx.Document(file_path)
        full_text = "\n".join([para.text for para in document.paragraphs])
        return [Page(page_number=1, text=full_text, source="text")]
    except Exception as e:
        logging.error(f"Failed to parse DOCX '{file_path.name}': {e}", exc_info=True)
        return []

def _parse_html(file_path: Path) -> List[Page]:
    """Parses an .html document. Lacks pagination, so returns as a single page."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            return [Page(page_number=1, text=soup.get_text(), source="text")]
    except Exception as e:
        logging.error(f"Failed to parse HTML '{file_path.name}': {e}", exc_info=True)
        return []

def _parse_txt(file_path: Path) -> List[Page]:
    """Parses a .txt document. Lacks pagination, so returns as a single page."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [Page(page_number=1, text=f.read(), source="text")]
    except Exception as e:
        logging.error(f"Failed to parse TXT '{file_path.name}': {e}", exc_info=True)
        return []

# --- Main Parser Dispatcher ---

PARSERS: Dict[str, Callable[[Path], List[Page]]] = {
    ".pdf": _parse_pdf,
    ".docx": _parse_docx,
    ".html": _parse_html,
    ".htm": _parse_html,
    ".txt": _parse_txt,
}

def parse_document(file_path: str | Path) -> List[Page]:
    """
    Parses a document from the given file path by dispatching to the
    appropriate parser based on the file extension.

    Args:
        file_path: The path to the document file.

    Returns:
        A list of Page objects, each containing the text of a page.

    Raises:
        ValueError: If the file type is not supported.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"The file was not found: {path}")

    ext = path.suffix.lower()
    parser = PARSERS.get(ext)

    if parser is None:
        raise ValueError(f"Unsupported file type: '{ext}' for file '{path.name}'")

    logging.info(f"Parsing document '{path.name}' with '{parser.__name__}'...")
    return parser(path)

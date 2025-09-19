"""
Main entry point that orchestrates the entire extraction pipeline.
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import argparse

# Import project modules
from config import settings
from utils import setup_logging, timer_decorator
from excel_handler import ExcelHandler
from pdf_processor import PDFProcessor
from embedding_manager import EmbeddingManager
from llm_client import LLMClient
from feature_extractor import FeatureExtractor
from yaml_handler import YAMLHandler

# Setup logging
setup_logging(settings.log_level, settings.log_file)
logger = logging.getLogger(__name__)

@timer_decorator
def process_single_paper(pdf_path: Path, fields: List[Dict[str, Any]], excel_handler: ExcelHandler,
                         embedding_manager: EmbeddingManager, feature_extractor: FeatureExtractor):
    """
    Processes a single PDF paper: embedding, feature extraction, and saving results.
    """
    paper_name = pdf_path.name
    paper_id = pdf_path.stem  # Use filename without extension as unique ID
    logger.info(f"--- Starting processing for paper: {paper_name} (ID: {paper_id}) ---")

    try:
        # Get or create the vector collection for the paper. Caching is handled here.
        paper_collection = embedding_manager.get_or_create_collection(paper_id, str(pdf_path))
        if not paper_collection:
            logger.error(f"Could not create or retrieve a vector collection for {paper_name}. Skipping.")
            return

        # Extract all features from the paper
        extracted_data = feature_extractor.extract_all_features(paper_collection, fields)

        # Save results progressively
        processing_time = time.time()  # Will be recalculated by the decorator
        excel_handler.update_results(settings.results_file, paper_name, extracted_data, processing_time)

        logger.info(f"--- Finished processing for paper: {paper_name} ---")

    except Exception as e:
        logger.critical(f"A critical error occurred while processing {paper_name}: {e}", exc_info=True)
        # Optionally save error state to Excel
        field_names = [field['field_name'] for field in fields]
        error_data = {feature: {"value": "PROCESSING_ERROR", "confidence": 0, "found": False} for feature in field_names}
        excel_handler.update_results(settings.results_file, paper_name, error_data, 0)


def process_all_papers(fields: List[Dict[str, Any]], excel_handler: ExcelHandler,
                       embedding_manager: EmbeddingManager, feature_extractor: FeatureExtractor,
                       force_reprocess: bool, paper_list: List[str] = None):
    """
    Scans the papers folder and processes each PDF.
    """
    papers_dir = settings.papers_folder
    if paper_list:
        pdf_files = [papers_dir / paper for paper in paper_list if (papers_dir / paper).exists()]
    else:
        pdf_files = list(papers_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in directory: {papers_dir}")
        return

    logger.info(f"Found {len(pdf_files)} PDF(s) to process.")

    # Initialize results file with headers
    field_names = [field['field_name'] for field in fields]
    excel_handler.initialize_results_file(settings.results_file, field_names)

    # If force_reprocess is enabled, clear existing collections
    if force_reprocess:
        logger.info("Force reprocess enabled. Deleting existing collections for selected papers.")
        for pdf_path in pdf_files:
            paper_id = pdf_path.stem
            embedding_manager.delete_collection(paper_id)

    for pdf_path in tqdm(pdf_files, desc="Processing Papers"):
        process_single_paper(pdf_path, fields, excel_handler, embedding_manager, feature_extractor)

def main(args):
    """
    Main execution flow.
    """
    logger.info("--- Starting PDF Feature Extraction System ---")

    # Override settings with CLI args if provided
    if args.results:
        settings.results_file = Path(args.results)
    if args.papers_folder:
        settings.papers_folder = Path(args.papers_folder)
        if not settings.papers_folder.exists():
            logger.critical(f"Papers folder specified via CLI does not exist: {settings.papers_folder}")
            return

    # 1. Initialize handlers and processors
    yaml_handler = YAMLHandler()
    excel_handler = ExcelHandler()
    pdf_processor = PDFProcessor(settings.chunk_size, settings.chunk_overlap)
    embedding_manager = EmbeddingManager(settings.chroma_db_path, settings.embedding_model, pdf_processor)

    try:
        llm_client = LLMClient(
            provider=settings.llm_provider,
            base_url=settings.ollama_base_url if settings.llm_provider == 'ollama' else settings.vllm_base_url,
            model_name=settings.model_name,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens
        )
    except ValueError as e:
        logger.critical(f"Failed to initialize LLM Client: {e}. Aborting.")
        return

    feature_extractor = FeatureExtractor(embedding_manager, llm_client, settings.top_k_chunks)

    # 2. Load field configurations from YAML
    try:
        fields = yaml_handler.load_fields(settings.fields_config_file)
    except (FileNotFoundError, ValueError) as e:
        logger.critical(f"Failed to load or parse fields config file: {e}. Aborting.")
        return

    # 3. Process all papers
    if not args.dry_run:
        process_all_papers(
            fields=fields,
            excel_handler=excel_handler,
            embedding_manager=embedding_manager,
            feature_extractor=feature_extractor,
            force_reprocess=args.force,
            paper_list=args.papers
        )
    else:
        logger.info("--- Dry run mode enabled. No results will be saved. ---")

    # 4. Optional: Clean up resources
    if args.clear_cache:
        logger.info("Clearing ALL ChromaDB collections as requested.")
        collections = embedding_manager.list_collections()
        for collection_name in tqdm(collections, desc="Deleting All Collections"):
            embedding_manager.delete_collection(collection_name)

    logger.info("--- PDF Feature Extraction System Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Academic Paper Feature Extraction System")
    parser.add_argument("--papers-folder", type=str, help=f"Directory containing PDF files. Overrides .env setting: {settings.papers_folder}")
    parser.add_argument("--results", type=str, help=f"Path to save the results Excel/CSV file. Overrides .env setting: {settings.results_file}")
    parser.add_argument("--papers", nargs='+', help="Process only specific paper filenames from the papers folder.")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of papers by deleting and rebuilding their vector collections.")
    parser.add_argument("--dry-run", action="store_true", help="Run the process without saving any results to the Excel file.")
    parser.add_argument("--clear-cache", action="store_true", help="Clear all cached paper embeddings from ChromaDB after the run.")

    cli_args = parser.parse_args()
    main(cli_args)

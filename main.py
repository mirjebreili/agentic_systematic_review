"""
Main entry point that orchestrates the entire extraction pipeline.
"""

import logging
import time
from pathlib import Path
from typing import List
from tqdm import tqdm
import argparse
import pandas as pd

# Import project modules
from config import settings
from utils import setup_logging, generate_paper_id, timer_decorator
from excel_handler import ExcelHandler
from pdf_processor import PDFProcessor
from embedding_manager import EmbeddingManager
from llm_client import LLMClient
from feature_extractor import FeatureExtractor

# Setup logging
setup_logging(settings.log_level, settings.log_file)
logger = logging.getLogger(__name__)

@timer_decorator
def process_single_paper(pdf_path: Path, features_df: pd.DataFrame, excel_handler: ExcelHandler,
                         embedding_manager: EmbeddingManager, feature_extractor: FeatureExtractor,
                         force_reprocess: bool = False):
    """
    Processes a single PDF paper: embedding, feature extraction, and saving results.
    """
    paper_name = pdf_path.name
    paper_id = generate_paper_id(str(pdf_path))
    logger.info(f"--- Starting processing for paper: {paper_name} (ID: {paper_id}) ---")

    # Check if paper should be skipped (if not forcing re-processing)
    if not force_reprocess:
        try:
            results_df = pd.read_csv(settings.results_file) if str(settings.results_file).endswith('.csv') else pd.read_excel(settings.results_file)
            if paper_name in results_df["Paper_Name"].values:
                logger.info(f"Paper '{paper_name}' already found in results. Skipping.")
                return
        except FileNotFoundError:
            pass # Results file doesn't exist yet, so we can't skip anything

    try:
        # Get or create the vector collection for the paper
        paper_collection = embedding_manager.get_or_create_collection(paper_id, str(pdf_path))
        if not paper_collection:
            logger.error(f"Could not create or retrieve a vector collection for {paper_name}. Skipping.")
            return

        # Extract all features from the paper
        extracted_data = feature_extractor.extract_all_features(paper_collection, features_df)

        # Save results progressively
        processing_time = time.time() # This will be recalculated by the decorator, but good enough for now
        excel_handler.update_results(settings.results_file, paper_name, extracted_data, processing_time)

        logger.info(f"--- Finished processing for paper: {paper_name} ---")

    except Exception as e:
        logger.critical(f"A critical error occurred while processing {paper_name}: {e}", exc_info=True)
        # Optionally save error state to Excel
        error_data = {feature: {"value": "PROCESSING_ERROR", "confidence": 0, "found": False} for feature in features_df["Feature_Name"]}
        excel_handler.update_results(settings.results_file, paper_name, error_data, 0)


def process_all_papers(pdf_dir: Path, features_df: pd.DataFrame, excel_handler: ExcelHandler,
                       embedding_manager: EmbeddingManager, feature_extractor: FeatureExtractor,
                       force_reprocess: bool, paper_list: List[str] = None):
    """
    Scans the PDF directory and processes each paper.
    """
    if paper_list:
        pdf_files = [pdf_dir / paper for paper in paper_list]
    else:
        pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in directory: {pdf_dir}")
        return

    logger.info(f"Found {len(pdf_files)} PDF(s) to process.")

    # Initialize results file with headers
    feature_names = features_df["Feature_Name"].tolist()
    excel_handler.initialize_results_file(settings.results_file, feature_names)

    for pdf_path in tqdm(pdf_files, desc="Processing Papers"):
        process_single_paper(pdf_path, features_df, excel_handler, embedding_manager, feature_extractor, force_reprocess)

def main(args):
    """
    Main execution flow.
    """
    logger.info("--- Starting PDF Feature Extraction System ---")

    # Override settings with CLI args if provided
    if args.pdf_dir:
        settings.pdf_directory = Path(args.pdf_dir)
    if args.features:
        settings.features_file = Path(args.features)
    if args.results:
        settings.results_file = Path(args.results)

    # 1. Initialize handlers and processors
    excel_handler = ExcelHandler()
    pdf_processor = PDFProcessor(settings.chunk_size, settings.chunk_overlap)
    embedding_manager = EmbeddingManager(settings.chroma_db_path, settings.embedding_model, pdf_processor)

    # Conditionally initialize LLMClient based on connectivity check in settings
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

    # 2. Load features from Excel
    try:
        features_df = excel_handler.load_features(settings.features_file)
    except (FileNotFoundError, ValueError) as e:
        logger.critical(f"Failed to load features file: {e}. Aborting.")
        return

    # 3. Process all papers
    if not args.dry_run:
        process_all_papers(
            pdf_dir=settings.pdf_directory,
            features_df=features_df,
            excel_handler=excel_handler,
            embedding_manager=embedding_manager,
            feature_extractor=feature_extractor,
            force_reprocess=args.force,
            paper_list=args.papers
        )
    else:
        logger.info("--- Dry run mode enabled. No results will be saved. ---")
        # You could add logic here to simulate a run without writing files

    # 4. Optional: Clean up resources
    if args.clear_cache:
        logger.info("Clearing ChromaDB cache as requested.")
        collections = embedding_manager.list_collections()
        for collection_name in tqdm(collections, desc="Deleting Collections"):
            embedding_manager.delete_collection(collection_name)

    logger.info("--- PDF Feature Extraction System Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Academic Paper Feature Extraction System")
    parser.add_argument("--pdf-dir", type=str, help=f"Directory containing PDF files. Overrides .env setting: {settings.pdf_directory}")
    parser.add_argument("--features", type=str, help=f"Path to the features Excel/CSV file. Overrides .env setting: {settings.features_file}")
    parser.add_argument("--results", type=str, help=f"Path to save the results Excel/CSV file. Overrides .env setting: {settings.results_file}")
    parser.add_argument("--papers", nargs='+', help="Process only specific paper filenames from the PDF directory.")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of papers that are already in the results file.")
    parser.add_argument("--dry-run", action="store_true", help="Run the process without saving any results to the Excel file.")
    parser.add_argument("--clear-cache", action="store_true", help="Clear all cached paper embeddings from ChromaDB after the run.")

    cli_args = parser.parse_args()
    main(cli_args)

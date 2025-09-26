"""Main entry point for orchestrating the agent-driven extraction pipeline."""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import Dict, List

from tqdm import tqdm

from agents import EmbeddingAgent, ExtractionAgent, PDFAgent, ResultsAgent
from config import settings
from embedding_manager import EmbeddingManager
from excel_handler import ExcelHandler
from feature_extractor import FeatureExtractor
from llm_client import LLMClient
from pdf_processor import PDFProcessor
from utils import setup_logging
from yaml_handler import YAMLHandler

# Configure logging as early as possible.
setup_logging(settings.log_level, settings.log_file)
logger = logging.getLogger(__name__)


def _resolve_pdf_files(paper_list: List[str] | None) -> List[Path]:
    """Collect and validate the PDF files to process."""
    papers_dir = settings.papers_folder
    if paper_list:
        resolved_files: List[Path] = []
        for paper in paper_list:
            candidate = papers_dir / paper
            if candidate.exists():
                resolved_files.append(candidate)
            else:
                logger.warning("Requested paper %s was not found in %s", paper, papers_dir)
        return resolved_files

    return list(papers_dir.glob("*.pdf"))


def _process_single_pdf(
    pdf_path: Path,
    fields: List[Dict[str, object]],
    field_names: List[str],
    excel_handler: ExcelHandler,
    embedding_manager: EmbeddingManager,
    feature_extractor: FeatureExtractor,
) -> List[Dict[str, object]]:
    """Run the agent pipeline for a single PDF and return summary messages."""
    message_queue: Queue = Queue()

    pdf_agent = PDFAgent(message_queue, [pdf_path])
    embedding_agent = EmbeddingAgent(message_queue, embedding_manager)
    extraction_agent = ExtractionAgent(message_queue, feature_extractor, fields)
    results_agent = ResultsAgent(message_queue, excel_handler, settings.results_file, field_names)

    try:
        pdf_agent.run()
        embedding_agent.run()
        extraction_agent.run()
        results_agent.run()
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Pipeline crashed for %s: %s", pdf_path, exc, exc_info=True)
        return [
            {
                "status": "error",
                "stage": "results_done",
                "paper_id": pdf_path.stem,
                "paper_name": pdf_path.name,
                "payload": {"error": str(exc)},
            }
        ]

    summary_messages: List[Dict[str, object]] = []
    while True:
        message = message_queue.get()
        if message.get("type") == "sentinel" and message.get("stage") == "results_done":
            break
        summary_messages.append(message)

    return summary_messages


def _chunk_pdfs(pdf_files: List[Path], batch_size: int) -> List[List[Path]]:
    """Split PDF paths into batches respecting the configured batch size."""
    batch_size = max(batch_size, 1)
    return [pdf_files[i : i + batch_size] for i in range(0, len(pdf_files), batch_size)]


def _run_pipeline(
    pdf_files: List[Path],
    fields: List[Dict[str, object]],
    excel_handler: ExcelHandler,
    embedding_manager: EmbeddingManager,
    feature_extractor: FeatureExtractor,
) -> None:
    """Execute the agent pipeline for PDFs in batches with bounded concurrency."""
    if not pdf_files:
        logger.warning("No PDF files found in directory: %s", settings.papers_folder)
        return

    field_names = [field["field_name"] for field in fields]
    excel_handler.initialize_results_file(settings.results_file, field_names)

    batches = _chunk_pdfs(list(pdf_files), settings.batch_size)
    total_batches = len(batches)

    summary_messages: List[Dict[str, object]] = []

    for batch_index, batch in enumerate(batches, start=1):
        logger.info(
            "Starting batch %d/%d with %d papers", batch_index, total_batches, len(batch)
        )
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_pdf = {
                executor.submit(
                    _process_single_pdf,
                    pdf_path,
                    fields,
                    field_names,
                    excel_handler,
                    embedding_manager,
                    feature_extractor,
                ): pdf_path
                for pdf_path in batch
            }

            for future in as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                try:
                    summary_messages.extend(future.result())
                except Exception as exc:  # pylint: disable=broad-except
                    logger.error(
                        "Unhandled error while processing %s: %s", pdf_path, exc, exc_info=True
                    )
                    summary_messages.append(
                        {
                            "status": "error",
                            "stage": "results_done",
                            "paper_id": pdf_path.stem,
                            "paper_name": pdf_path.name,
                            "payload": {"error": str(exc)},
                        }
                    )

        logger.info("Completed batch %d/%d", batch_index, total_batches)

    for message in summary_messages:
        if message.get("stage") != "results_done":
            continue
        paper = message.get("paper_name") or message.get("paper_id")
        if message.get("status") == "success":
            processing_time = message.get("payload", {}).get("processing_time", 0)
            logger.info("Successfully processed %s in %.2fs", paper, processing_time)
        else:
            error_detail = message.get("payload", {}).get("error", "Unknown error")
            logger.error("Processing failed for %s: %s", paper, error_detail)


def main(args: argparse.Namespace) -> None:
    """Main execution flow."""
    logger.info("--- Starting PDF Feature Extraction System ---")

    if args.results:
        settings.results_file = Path(args.results)
    if args.papers_folder:
        settings.papers_folder = Path(args.papers_folder)
        if not settings.papers_folder.exists():
            logger.critical("Papers folder specified via CLI does not exist: %s", settings.papers_folder)
            return

    yaml_handler = YAMLHandler()
    excel_handler = ExcelHandler()
    pdf_processor = PDFProcessor(settings.chunk_size, settings.chunk_overlap)
    embedding_manager = EmbeddingManager(settings.chroma_db_path, settings.embedding_model, pdf_processor)

    try:
        llm_client = LLMClient(
            provider=settings.llm_provider,
            base_url=settings.ollama_base_url if settings.llm_provider == "ollama" else settings.vllm_base_url,
            model_name=settings.model_name,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
    except ValueError as exc:
        logger.critical("Failed to initialize LLM Client: %s. Aborting.", exc)
        return

    feature_extractor = FeatureExtractor(embedding_manager, llm_client, settings.top_k_chunks)

    try:
        fields = yaml_handler.load_fields(settings.fields_config_file)
    except (FileNotFoundError, ValueError) as exc:
        logger.critical("Failed to load or parse fields config file: %s. Aborting.", exc)
        return

    pdf_files = _resolve_pdf_files(args.papers)

    if args.force:
        logger.info("Force reprocess enabled. Deleting existing collections for selected papers.")
        for pdf_path in pdf_files:
            embedding_manager.delete_collection(pdf_path.stem)

    if args.dry_run:
        logger.info("--- Dry run mode enabled. No results will be saved. ---")
    else:
        _run_pipeline(pdf_files, fields, excel_handler, embedding_manager, feature_extractor)

    if args.clear_cache:
        logger.info("Clearing ALL ChromaDB collections as requested.")
        for collection_name in tqdm(embedding_manager.list_collections(), desc="Deleting All Collections"):
            embedding_manager.delete_collection(collection_name)

    logger.info("--- PDF Feature Extraction System Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Academic Paper Feature Extraction System")
    parser.add_argument(
        "--papers-folder",
        type=str,
        help=f"Directory containing PDF files. Overrides .env setting: {settings.papers_folder}",
    )
    parser.add_argument(
        "--results",
        type=str,
        help=f"Path to save the results Excel/CSV file. Overrides .env setting: {settings.results_file}",
    )
    parser.add_argument("--papers", nargs="+", help="Process only specific paper filenames from the papers folder.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of papers by deleting and rebuilding their vector collections.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the process without saving any results to the Excel file.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all cached paper embeddings from ChromaDB after the run.",
    )

    main(parser.parse_args())


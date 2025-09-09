"""
The Exporter Agent gathers all results and materializes the final outputs,
including reports, logs, and copies of included papers.
"""
import logging
import json
import shutil
from pathlib import Path
import pandas as pd
from typing import Dict, List

from src.schemas.output_schemas import AppState, DocumentExtractionResult, ExtractedDataPoint, ScreeningResult

def export_results(state: Dict) -> Dict:
    """
    The entry point for the Exporter agent.

    This agent is responsible for writing all the results of the pipeline
    to disk in a structured, timestamped output directory.

    Args:
        state: The final application state dictionary.

    Returns:
        The application state dictionary, unchanged.
    """
    app_state = AppState(**state)
    output_path = Path(app_state.output_path)

    logging.info(f"--- Starting Export Stage ---")
    logging.info(f"Writing all outputs to directory: {output_path}")

    # Ensure the main output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Copy included papers
    _export_included_papers(app_state, output_path)

    # 2. Write screening_report.jsonl
    _export_screening_report(app_state.screening_results, output_path)

    # 3. Write extraction_results.xlsx and provenance_log.csv
    _export_extraction_results(app_state.extraction_results, output_path)

    # 4. Write run_summary.md
    _export_run_summary(app_state, output_path)

    # 5. Write errors.log
    _export_error_log(app_state.processing_errors, output_path)

    logging.info("--- Export Stage Complete ---")
    return app_state.model_dump()

def _export_included_papers(app_state: AppState, output_path: Path):
    included_dir = output_path / "included_papers"
    included_dir.mkdir(exist_ok=True)

    included_doc_ids = {res.document_id for res in app_state.screening_results if res.status == "Include"}

    # Create a map of doc_id to full path from the initial list
    doc_path_map = {Path(p).name: p for p in app_state.documents_to_process}

    if not included_doc_ids:
        logging.info("No papers marked for inclusion. 'included_papers' directory will be empty.")
        return

    copied_count = 0
    for doc_id in included_doc_ids:
        if doc_id in doc_path_map:
            source_path = Path(doc_path_map[doc_id])
            dest_path = included_dir / doc_id
            try:
                shutil.copy(source_path, dest_path)
                copied_count += 1
            except Exception as e:
                logging.error(f"Failed to copy included paper '{doc_id}' from '{source_path}' to '{dest_path}': {e}")
                app_state.processing_errors[f"export_copy_{doc_id}"] = str(e)
    logging.info(f"Copied {copied_count} included papers to '{included_dir}'.")

def _export_screening_report(results: List[ScreeningResult], output_path: Path):
    report_path = output_path / "screening_report.jsonl"
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            for res in results:
                f.write(res.model_dump_json() + '\n')
        logging.info(f"Screening report saved to '{report_path}'.")
    except Exception as e:
        logging.error(f"Failed to write screening report: {e}")

def _export_extraction_results(results: List[DocumentExtractionResult], output_path: Path):
    if not results:
        logging.info("No extraction results to export. Skipping Excel and CSV reports.")
        return

    # Create provenance log (long format)
    provenance_records = []
    for res in results:
        for dp in res.extracted_features:
            record = dp.model_dump()
            record['document_id'] = res.document_id
            provenance_records.append(record)

    provenance_df = pd.DataFrame(provenance_records)
    provenance_path = output_path / "provenance_log.csv"
    provenance_df.to_csv(provenance_path, index=False, encoding='utf-8')
    logging.info(f"Provenance log saved to '{provenance_path}'.")

    # Create main Excel report (wide format)
    excel_records = []
    for res in results:
        record = {'document_id': res.document_id}
        for dp in res.extracted_features:
            record[dp.feature_name] = dp.value
        excel_records.append(record)

    excel_df = pd.DataFrame(excel_records)
    excel_path = output_path / "extraction_results.xlsx"
    excel_df.to_excel(excel_path, index=False)
    logging.info(f"Main extraction report saved to '{excel_path}'.")

def _export_run_summary(app_state: AppState, output_path: Path):
    summary_path = output_path / "run_summary.md"

    total_docs = len(app_state.documents_to_process)
    included = sum(1 for r in app_state.screening_results if r.status == "Include")
    excluded = sum(1 for r in app_state.screening_results if r.status == "Exclude")
    unsure = sum(1 for r in app_state.screening_results if r.status == "Unsure")

    run_duration = "N/A"
    if app_state.start_time and app_state.end_time:
        duration_seconds = (app_state.end_time - app_state.start_time).total_seconds()
        run_duration = f"{duration_seconds:.2f} seconds"

    summary = f"""
# Run Summary

## Overview
- **Run Start Time:** {app_state.start_time}
- **Run End Time:** {app_state.end_time}
- **Total Duration:** {run_duration}
- **LLM Model Used:** `{app_state.model_name}`
- **Output Directory:** `{output_path}`

## Document Processing
- **Total Documents Found:** {total_docs}
- **Successfully Processed (Ingested):** {len(app_state.document_chunks)}
- **Processing Errors:** {len(app_state.processing_errors)}

## Screening Results
- **Included:** {included}
- **Excluded:** {excluded}
- **Unsure:** {unsure}

## Extraction Results
- **Documents with Extracted Features:** {len(app_state.extraction_results)}
"""
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        logging.info(f"Run summary saved to '{summary_path}'.")
    except Exception as e:
        logging.error(f"Failed to write run summary: {e}")

def _export_error_log(errors: Dict[str, str], output_path: Path):
    log_path = output_path / "errors.log"
    if not errors:
        logging.info("No processing errors were recorded.")
        return

    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            for source, error_msg in errors.items():
                f.write(f"[{source}]: {error_msg}\n")
        logging.info(f"Error log saved to '{log_path}'.")
    except Exception as e:
        logging.error(f"Failed to write error log: {e}")

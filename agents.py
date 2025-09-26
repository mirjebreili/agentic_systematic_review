"""Agent classes orchestrating the PDF processing pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Sequence


class PDFAgent:
    """Agent responsible for preparing PDF metadata for downstream stages."""

    def __init__(self, queue: Queue, pdf_files: Sequence[Path]):
        self.queue = queue
        self.pdf_files = list(pdf_files)
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self) -> None:
        """Enqueue PDF metadata for processing."""
        for pdf_path in self.pdf_files:
            try:
                if not pdf_path.exists():
                    raise FileNotFoundError(f"PDF not found at {pdf_path}")

                start_time = time.time()
                paper_name = pdf_path.name
                paper_id = pdf_path.stem

                message = {
                    "status": "success",
                    "stage": "pdf_processed",
                    "paper_id": paper_id,
                    "paper_name": paper_name,
                    "payload": {
                        "pdf_path": pdf_path,
                        "start_time": start_time,
                    },
                }
                self.queue.put(message)
                self.logger.debug("Queued PDF for processing: %s", pdf_path)
            except Exception as exc:  # pylint: disable=broad-except
                error_message = {
                    "status": "error",
                    "stage": "pdf_processed",
                    "paper_id": pdf_path.stem if isinstance(pdf_path, Path) else None,
                    "paper_name": pdf_path.name if isinstance(pdf_path, Path) else str(pdf_path),
                    "payload": {"error": str(exc)},
                }
                self.queue.put(error_message)
                self.logger.error("Failed to queue PDF %s: %s", pdf_path, exc, exc_info=True)

        self.queue.put({"type": "sentinel", "stage": "pdf_processed"})


class EmbeddingAgent:
    """Agent that manages embedding collection creation or retrieval."""

    def __init__(self, queue: Queue, embedding_manager: Any):
        self.queue = queue
        self.embedding_manager = embedding_manager
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self) -> None:
        """Consume PDF metadata messages and enqueue embedding results."""
        while True:
            message = self.queue.get()
            if message.get("type") == "sentinel" and message.get("stage") == "pdf_processed":
                self.queue.put({"type": "sentinel", "stage": "embedding_done"})
                break

            if message.get("status") != "success":
                self._forward_error(message, "embedding_done")
                continue

            try:
                pdf_path = Path(message["payload"]["pdf_path"])
                paper_id = message["paper_id"]
                collection = self.embedding_manager.get_or_create_collection(paper_id, str(pdf_path))
                if not collection:
                    raise RuntimeError(f"Unable to prepare embedding collection for {paper_id}")

                enriched_message = {
                    "status": "success",
                    "stage": "embedding_done",
                    "paper_id": paper_id,
                    "paper_name": message.get("paper_name"),
                    "payload": {
                        "collection": collection,
                        "pdf_path": pdf_path,
                        "start_time": message["payload"].get("start_time"),
                    },
                }
                self.queue.put(enriched_message)
                self.logger.debug("Embedding ready for paper %s", paper_id)
            except Exception as exc:  # pylint: disable=broad-except
                error_message = {
                    "status": "error",
                    "stage": "embedding_done",
                    "paper_id": message.get("paper_id"),
                    "paper_name": message.get("paper_name"),
                    "payload": {"error": str(exc)},
                }
                self.queue.put(error_message)
                self.logger.error(
                    "Failed during embedding stage for %s: %s", message.get("paper_id"), exc, exc_info=True
                )

    def _forward_error(self, message: Dict[str, Any], stage: str) -> None:
        payload = message.get("payload", {})
        forwarded_message = {
            "status": "error",
            "stage": stage,
            "paper_id": message.get("paper_id"),
            "paper_name": message.get("paper_name"),
            "payload": payload if isinstance(payload, dict) else {"error": str(payload)},
        }
        self.queue.put(forwarded_message)


class ExtractionAgent:
    """Agent that runs feature extraction for embedded documents."""

    def __init__(self, queue: Queue, feature_extractor: Any, fields: Sequence[Dict[str, Any]]):
        self.queue = queue
        self.feature_extractor = feature_extractor
        self.fields = list(fields)
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self) -> None:
        """Consume embedding messages and enqueue extraction results."""
        while True:
            message = self.queue.get()
            if message.get("type") == "sentinel" and message.get("stage") == "embedding_done":
                self.queue.put({"type": "sentinel", "stage": "extraction_done"})
                break

            if message.get("status") != "success":
                self._forward_error(message, "extraction_done")
                continue

            try:
                collection = message["payload"]["collection"]
                extracted_data = self.feature_extractor.extract_all_features(collection, self.fields)
                enriched_message = {
                    "status": "success",
                    "stage": "extraction_done",
                    "paper_id": message.get("paper_id"),
                    "paper_name": message.get("paper_name"),
                    "payload": {
                        "extracted_data": extracted_data,
                        "start_time": message["payload"].get("start_time"),
                    },
                }
                self.queue.put(enriched_message)
                self.logger.debug("Extraction complete for paper %s", message.get("paper_id"))
            except Exception as exc:  # pylint: disable=broad-except
                error_message = {
                    "status": "error",
                    "stage": "extraction_done",
                    "paper_id": message.get("paper_id"),
                    "paper_name": message.get("paper_name"),
                    "payload": {"error": str(exc)},
                }
                self.queue.put(error_message)
                self.logger.error(
                    "Feature extraction failed for %s: %s", message.get("paper_id"), exc, exc_info=True
                )

    def _forward_error(self, message: Dict[str, Any], stage: str) -> None:
        payload = message.get("payload", {})
        forwarded_message = {
            "status": "error",
            "stage": stage,
            "paper_id": message.get("paper_id"),
            "paper_name": message.get("paper_name"),
            "payload": payload if isinstance(payload, dict) else {"error": str(payload)},
        }
        self.queue.put(forwarded_message)


class ResultsAgent:
    """Agent that persists extraction results to the configured output store."""

    def __init__(
        self,
        queue: Queue,
        excel_handler: Any,
        results_file: Path,
        field_names: Sequence[str],
    ) -> None:
        self.queue = queue
        self.excel_handler = excel_handler
        self.results_file = results_file
        self.field_names = list(field_names)
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self) -> None:
        """Consume extraction results and persist them."""
        while True:
            message = self.queue.get()
            if message.get("type") == "sentinel" and message.get("stage") == "extraction_done":
                self.queue.put({"type": "sentinel", "stage": "results_done"})
                break

            paper_name = message.get("paper_name") or message.get("paper_id") or "unknown"
            try:
                if message.get("status") == "success":
                    payload = message.get("payload", {})
                    extracted_data = payload.get("extracted_data", {})
                    start_time = payload.get("start_time")
                    processing_time = time.time() - start_time if start_time else 0

                    self.excel_handler.update_results(
                        self.results_file,
                        paper_name,
                        extracted_data,
                        processing_time,
                    )

                    success_message = {
                        "status": "success",
                        "stage": "results_done",
                        "paper_id": message.get("paper_id"),
                        "paper_name": paper_name,
                        "payload": {"processing_time": processing_time},
                    }
                    self.queue.put(success_message)
                    self.logger.debug("Results recorded for paper %s", paper_name)
                else:
                    payload = message.get("payload", {})
                    error_reason = payload.get("error", "Unknown error")
                    error_data = {
                        field: {"value": "PROCESSING_ERROR", "confidence": 0, "found": False}
                        for field in self.field_names
                    }

                    self.excel_handler.update_results(self.results_file, paper_name, error_data, 0)

                    error_message = {
                        "status": "error",
                        "stage": "results_done",
                        "paper_id": message.get("paper_id"),
                        "paper_name": paper_name,
                        "payload": {"error": error_reason},
                    }
                    self.queue.put(error_message)
                    self.logger.warning("Recorded error for paper %s: %s", paper_name, error_reason)
            except Exception as exc:  # pylint: disable=broad-except
                failure_message = {
                    "status": "error",
                    "stage": "results_done",
                    "paper_id": message.get("paper_id"),
                    "paper_name": paper_name,
                    "payload": {"error": str(exc)},
                }
                self.queue.put(failure_message)
                self.logger.error(
                    "Failed to persist results for paper %s: %s", paper_name, exc, exc_info=True
                )


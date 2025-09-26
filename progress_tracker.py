"""Utilities for tracking paper processing progress with atomic persistence."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class ProgressTracker:
    """Manage loading and persisting paper processing progress."""

    def __init__(self, progress_file: Path) -> None:
        self.progress_file = progress_file
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(__name__)
        self._progress: Dict[str, Dict[str, Any]] = {}
        self.refresh()

    def refresh(self) -> Dict[str, Dict[str, Any]]:
        """Reload the progress file from disk."""
        self._progress = self._load_progress()
        return self._progress

    def get_processed_papers(self) -> set[str]:
        """Return the set of paper identifiers recorded in progress."""
        return set(self._progress.keys())

    def record(
        self,
        paper_id: str,
        paper_name: str,
        status: str,
        details: Dict[str, Any] | None = None,
    ) -> None:
        """Persist the latest status for a paper using an atomic file replace."""
        entry: Dict[str, Any] = {
            "paper_name": paper_name,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if details:
            entry["details"] = details

        self._progress[paper_id] = entry
        self._write_progress(self._progress)

    def remove(self, paper_id: str) -> None:
        """Remove a paper from the progress tracking file."""
        if paper_id not in self._progress:
            return
        del self._progress[paper_id]
        self._write_progress(self._progress)

    def _load_progress(self) -> Dict[str, Dict[str, Any]]:
        if not self.progress_file.exists():
            return {}

        try:
            with self.progress_file.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError as exc:
            self._logger.warning(
                "Could not parse progress file %s: %s. Resetting progress state.",
                self.progress_file,
                exc,
            )
            return {}
        except OSError as exc:
            self._logger.warning(
                "Failed to read progress file %s: %s. Assuming no progress.",
                self.progress_file,
                exc,
            )
            return {}

        if not isinstance(data, dict):
            self._logger.warning(
                "Progress file %s contained unexpected data. Resetting progress state.",
                self.progress_file,
            )
            return {}

        # Ensure nested entries are dictionaries to avoid type issues later.
        sanitized: Dict[str, Dict[str, Any]] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                sanitized[key] = value
            else:
                sanitized[key] = {"status": str(value)}
        return sanitized

    def _write_progress(self, data: Dict[str, Dict[str, Any]]) -> None:
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=str(self.progress_file.parent), delete=False
            ) as tmp_file:
                json.dump(data, tmp_file, indent=2, sort_keys=True)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
                temp_path = Path(tmp_file.name)
        except OSError as exc:
            self._logger.error(
                "Failed to write progress file %s: %s", self.progress_file, exc
            )
            return

        try:
            temp_path.replace(self.progress_file)
        except OSError as exc:
            self._logger.error(
                "Failed to replace progress file %s: %s", self.progress_file, exc
            )
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            return

        # Ensure in-memory cache stays aligned with disk state.
        self._progress = data.copy()


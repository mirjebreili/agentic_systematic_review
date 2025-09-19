"""
Handle Excel input/output operations.
"""

import pandas as pd
from typing import Dict, List
from pathlib import Path
import logging
from filelock import FileLock
import datetime

logger = logging.getLogger(__name__)

class ExcelHandler:
    def initialize_results_file(self, results_path: Path, features: List[str]):
        """Create results Excel file with headers if it doesn't exist."""
        lock_path = results_path.with_suffix(f"{results_path.suffix}.lock")
        lock = FileLock(lock_path)
        with lock:
            if results_path.exists():
                logger.info(f"Results file already exists at {results_path}. Skipping initialization.")
                return

            headers = ["Paper_Name"] + features + ["extraction_date", "confidence_avg", "processing_time_seconds"]
            df = pd.DataFrame(columns=headers)

            try:
                if str(results_path).endswith('.csv'):
                    df.to_csv(results_path, index=False)
                else:
                    df.to_excel(results_path, index=False, engine='openpyxl')
                logger.info(f"Initialized empty results file at {results_path}")
            except Exception as e:
                logger.error(f"Failed to initialize results file: {e}")
                raise

    def update_results(self, results_path: Path, paper_name: str, extracted_data: Dict, processing_time: float):
        """
        Add or update a paper's results in the Excel/CSV file safely.
        """
        lock_path = results_path.with_suffix(f"{results_path.suffix}.lock")
        lock = FileLock(lock_path)

        with lock:
            try:
                if str(results_path).endswith('.csv'):
                    df = pd.read_csv(results_path)
                else:
                    df = pd.read_excel(results_path, engine='openpyxl')
            except FileNotFoundError:
                logger.error(f"Results file '{results_path}' not found. It should have been initialized before processing starts.")
                # As a fallback, create it now, but this indicates a logic error in the main script.
                features = list(extracted_data.keys())
                self.initialize_results_file(results_path, features)
                if str(results_path).endswith('.csv'):
                    df = pd.read_csv(results_path)
                else:
                    df = pd.read_excel(results_path, engine='openpyxl')

            new_row = {"Paper_Name": paper_name}

            confidences = [item.get('confidence', 0) for item in extracted_data.values() if isinstance(item, dict)]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            for feature, result in extracted_data.items():
                if isinstance(result, dict):
                    new_row[feature] = result.get("value", "NOT_FOUND")
                else:
                    new_row[feature] = result

            new_row["extraction_date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_row["confidence_avg"] = round(avg_confidence, 2)
            new_row["processing_time_seconds"] = round(processing_time, 2)

            # Remove existing row for the same paper if it exists
            if paper_name in df["Paper_Name"].values:
                df = df[df["Paper_Name"] != paper_name]

            new_df = pd.DataFrame([new_row])
            df = pd.concat([df, new_df], ignore_index=True)

            try:
                if str(results_path).endswith('.csv'):
                    df.to_csv(results_path, index=False)
                else:
                    df.to_excel(results_path, index=False, engine='openpyxl')
                logger.info(f"Successfully updated results for {paper_name} in {results_path}")
            except Exception as e:
                logger.error(f"Failed to write updates to results file: {e}")
                raise

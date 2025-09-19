import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from pathlib import Path
import shutil

# It's important to set the root path for imports to work correctly
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import process_single_paper
from excel_handler import ExcelHandler
from pdf_processor import PDFProcessor
from embedding_manager import EmbeddingManager
from llm_client import LLMClient
from feature_extractor import FeatureExtractor
from config import settings
from langchain_core.documents import Document

class TestIntegration(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test artifacts
        self.test_dir = Path("./test_artifacts")
        self.test_dir.mkdir(exist_ok=True)
        self.chroma_path = self.test_dir / "chroma_db"
        self.results_path = self.test_dir / "results.csv"

        # Sample PDF and features
        self.sample_pdf_path = Path(__file__).parent / "sample.pdf"
        self.sample_features_path = Path(__file__).parent / "sample_features.csv"

    def tearDown(self):
        # Clean up test artifacts
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    @patch('main.settings.results_file', new_callable=lambda: Path("./test_artifacts/results.csv"))
    @patch('llm_client.LLMClient')
    @patch('pdf_processor.PDFPlumberLoader')
    def test_end_to_end_single_paper(self, mock_loader, mock_llm_client_class, mock_settings):
        # --- Mocks Setup ---
        # Mock PDF Loader
        mock_loader.return_value.load.return_value = [
            Document(page_content=self.sample_pdf_path.read_text(), metadata={"page": 1})
        ]

        # Mock LLM Client
        mock_llm_instance = mock_llm_client_class.return_value
        mock_llm_instance.extract_feature.side_effect = [
            {"value": 500, "confidence": 0.9, "found": True, "explanation": "..."},
            {"value": "reduction in blood pressure", "confidence": 0.8, "found": True, "explanation": "..."},
            {"value": "< 0.05", "confidence": 0.95, "found": True, "explanation": "..."},
            {"value": "National Institutes of Health", "confidence": 0.85, "found": True, "explanation": "..."}
        ]

        # --- Test Dependencies Initialization ---
        excel_handler = ExcelHandler()
        features_df = excel_handler.load_features(self.sample_features_path)

        pdf_processor = PDFProcessor(chunk_size=150, chunk_overlap=30)

        embedding_manager = EmbeddingManager(
            chroma_db_path=str(self.chroma_path),
            embedding_model="all-MiniLM-L6-v2",
            pdf_processor=pdf_processor
        )

        feature_extractor = FeatureExtractor(
            embedding_manager=embedding_manager,
            llm_client=mock_llm_instance,
            top_k_chunks=1
        )

        # Initialize results file before running
        excel_handler.initialize_results_file(self.results_path, features_df["Feature_Name"].tolist())

        # --- Run the main processing function ---
        process_single_paper(
            pdf_path=self.sample_pdf_path,
            features_df=features_df,
            excel_handler=excel_handler,
            embedding_manager=embedding_manager,
            feature_extractor=feature_extractor,
            force_reprocess=True
        )

        # --- Assertions ---
        self.assertTrue(self.results_path.exists())
        results_df = pd.read_csv(self.results_path)

        self.assertEqual(len(results_df), 1)
        self.assertEqual(results_df.iloc[0]["Paper_Name"], self.sample_pdf_path.name)
        self.assertEqual(results_df.iloc[0]["sample_size"], 500)
        self.assertEqual(results_df.iloc[0]["primary_outcome"], "reduction in blood pressure")
        self.assertEqual(results_df.iloc[0]["p_value"], "< 0.05")
        self.assertEqual(results_df.iloc[0]["funding_source"], "National Institutes of Health")

        self.assertTrue(self.chroma_path.exists())
        self.assertGreater(len(list(self.chroma_path.iterdir())), 0)


if __name__ == '__main__':
    unittest.main()

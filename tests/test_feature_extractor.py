import unittest
from unittest.mock import MagicMock
import pandas as pd
from feature_extractor import FeatureExtractor
from langchain_core.documents import Document

class TestFeatureExtractor(unittest.TestCase):

    def setUp(self):
        # Mock dependencies
        self.mock_embedding_manager = MagicMock()
        self.mock_llm_client = MagicMock()

        # Initialize the extractor with mocks
        self.feature_extractor = FeatureExtractor(
            embedding_manager=self.mock_embedding_manager,
            llm_client=self.mock_llm_client,
            top_k_chunks=3
        )

    def test_extract_single_feature_found(self):
        # Mock the behavior of dependencies
        mock_collection = MagicMock()
        self.mock_embedding_manager.semantic_search.return_value = [
            Document(page_content="The sample size was 100 participants.", metadata={"page": 5})
        ]
        self.mock_llm_client.extract_feature.return_value = {
            "value": 100,
            "confidence": 0.95,
            "found": True,
            "explanation": "Found in the text."
        }

        # Call the method
        result = self.feature_extractor.extract_single_feature(
            collection=mock_collection,
            feature_name="Sample Size",
            feature_description="The total number of participants."
        )

        # Assertions
        self.assertEqual(result["value"], 100)
        self.assertTrue(result["found"])
        self.mock_embedding_manager.semantic_search.assert_called_once()
        self.mock_llm_client.extract_feature.assert_called_once()

    def test_extract_single_feature_not_found(self):
        # Mock semantic search returning no chunks
        mock_collection = MagicMock()
        self.mock_embedding_manager.semantic_search.return_value = []

        # Call the method
        result = self.feature_extractor.extract_single_feature(
            collection=mock_collection,
            feature_name="Effect Size",
            feature_description="The measure of the treatment effect."
        )

        # Assertions
        self.assertEqual(result["value"], "NOT_FOUND")
        self.assertFalse(result["found"])
        # Ensure LLM was not called if no chunks were found
        self.mock_llm_client.extract_feature.assert_not_called()

    def test_extract_all_features(self):
        # Setup mocks and data
        mock_collection = MagicMock()
        features_df = pd.DataFrame([
            {"Feature_Name": "Sample Size", "Description": "Number of people."},
            {"Feature_Name": "P-Value", "Description": "The p-value."}
        ])

        # Mock the return value for each call to extract_single_feature
        self.mock_llm_client.extract_feature.side_effect = [
            {"value": 150, "confidence": 0.9, "found": True, "explanation": "..."},
            {"value": 0.05, "confidence": 0.8, "found": True, "explanation": "..."}
        ]
        self.mock_embedding_manager.semantic_search.return_value = [Document(page_content="...")]

        # Call the method
        results = self.feature_extractor.extract_all_features(mock_collection, features_df)

        # Assertions
        self.assertIn("Sample Size", results)
        self.assertIn("P-Value", results)
        self.assertEqual(results["Sample Size"]["value"], 150)
        self.assertEqual(results["P-Value"]["value"], 0.05)
        self.assertEqual(self.mock_llm_client.extract_feature.call_count, 2)

if __name__ == '__main__':
    unittest.main()

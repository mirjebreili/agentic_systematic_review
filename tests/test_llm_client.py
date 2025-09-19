import unittest
from unittest.mock import patch, MagicMock
from llm_client import LLMClient, FeatureExtractionOutput

class TestLLMClient(unittest.TestCase):

    @patch('llm_client.Ollama')
    def test_llm_client_initialization(self, mock_ollama):
        # Test if the client initializes correctly
        client = LLMClient(provider="ollama", base_url="http://localhost:11434", model_name="test-model", temperature=0.1, max_tokens=100)
        self.assertIsNotNone(client.llm)
        mock_ollama.assert_called_once()

    @patch('llm_client.Ollama')
    def test_extract_feature_mocked(self, mock_ollama):
        # Setup client
        client = LLMClient(provider="ollama", base_url="http://localhost:11434", model_name="test-model", temperature=0.1, max_tokens=100)

        # Create a mock for the final chain object
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "value": "Test Value",
            "confidence": 0.9,
            "found": True,
            "explanation": "Mocked explanation."
        }

        # The chain is constructed inside extract_feature as:
        # `chain = self.prompt_template | self.llm | self.output_parser`
        # We need to mock this construction. We can replace the `prompt_template`
        # on the instance with a mock that, when combined with other mocks,
        # returns our final `mock_chain`.

        mock_prompt_llm_chain = MagicMock()

        # client.llm is already a mock of Ollama
        # client.output_parser is a real object. We can leave it or mock it.

        # Let's mock the two __or__ calls
        client.prompt_template = MagicMock()
        client.prompt_template.__or__.return_value = mock_prompt_llm_chain
        mock_prompt_llm_chain.__or__.return_value = mock_chain

        # Call the method to be tested
        result = client.extract_feature("Test Feature", "A test description", "Some context.")

        # Assertions
        self.assertEqual(result["value"], "Test Value")
        self.assertEqual(result["confidence"], 0.9)
        self.assertTrue(result["found"])

        # Verify that the invoke method on our final mock chain was called
        mock_chain.invoke.assert_called_once()

        # Verify the arguments passed to invoke
        expected_invoke_args = {
            'feature_name': 'Test Feature',
            'feature_description': 'A test description',
            'context': 'Some context.'
        }
        mock_chain.invoke.assert_called_with(expected_invoke_args)

if __name__ == '__main__':
    unittest.main()

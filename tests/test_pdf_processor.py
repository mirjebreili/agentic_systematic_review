import unittest
from unittest.mock import patch, MagicMock
from pdf_processor import PDFProcessor
from langchain_core.documents import Document

class TestPDFProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = PDFProcessor(chunk_size=100, chunk_overlap=20)

    def test_create_smart_chunks(self):
        # Create mock documents
        docs = [
            Document(page_content="This is the first sentence. This is the second sentence.", metadata={"page": 1}),
            Document(page_content="This is the third sentence. This is the fourth sentence.", metadata={"page": 2}),
        ]

        paper_id = "test_paper"
        chunks = self.processor.create_smart_chunks(docs, paper_id)

        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)

        for chunk in chunks:
            self.assertIn("paper_id", chunk.metadata)
            self.assertEqual(chunk.metadata["paper_id"], paper_id)
            self.assertIn("chunk_index", chunk.metadata)
            self.assertIn("chunk_id", chunk.metadata)
            self.assertTrue(chunk.page_content)

if __name__ == '__main__':
    unittest.main()

import unittest
from unittest.mock import patch
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

    @patch.object(PDFProcessor, "create_smart_chunks")
    @patch.object(PDFProcessor, "extract_text_from_pdf")
    def test_extract_and_chunk_pdf_success(self, mock_extract, mock_create_chunks):
        mock_documents = [Document(page_content="content", metadata={"page": 1})]
        mock_chunks = [Document(page_content="chunk", metadata={"page": 1})]
        mock_extract.return_value = mock_documents
        mock_create_chunks.return_value = mock_chunks

        result = self.processor.extract_and_chunk_pdf("sample.pdf", "paper")

        self.assertEqual(result, mock_chunks)
        mock_extract.assert_called_once_with("sample.pdf")
        mock_create_chunks.assert_called_once_with(mock_documents, "paper")

    @patch.object(PDFProcessor, "extract_text_from_pdf", side_effect=ValueError("boom"))
    def test_extract_and_chunk_pdf_handles_extraction_error(self, _mock_extract):
        result = self.processor.extract_and_chunk_pdf("sample.pdf", "paper")
        self.assertIsNone(result)

    @patch.object(PDFProcessor, "create_smart_chunks", side_effect=RuntimeError("chunk boom"))
    @patch.object(PDFProcessor, "extract_text_from_pdf", return_value=[Document(page_content="content", metadata={"page": 1})])
    def test_extract_and_chunk_pdf_handles_chunking_error(self, mock_extract, _mock_chunks):
        result = self.processor.extract_and_chunk_pdf("sample.pdf", "paper")
        self.assertIsNone(result)
        mock_extract.assert_called_once_with("sample.pdf")

if __name__ == '__main__':
    unittest.main()

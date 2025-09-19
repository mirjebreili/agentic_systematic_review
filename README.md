# Agentic Systematic Review

An automated research paper analysis system that uses ChromaDB vector embeddings and Large Language Models to extract structured information from academic papers.

## 🎯 What This Project Does

This system automatically:
- Scans a folder of PDF research papers
- Creates separate vector databases for each paper using ChromaDB
- Extracts specific information fields (configurable via YAML)
- Outputs results to an Excel file with one row per paper

## 🛠️ Requirements

- Python 3.8+
- [Ollama](https://ollama.ai/) running locally
- At least 8GB RAM for embedding models

## 🚀 Quick Start

### 1. Clone and Install
```
git clone <repository-url>
cd agentic_systematic_review
pip install -r requirements.txt
```

### 2. Set Up Environment
```
# Copy example environment file
cp .env.example .env

# Edit .env with your settings (optional - defaults work fine)
```

### 3. Install and Start Ollama
```
# Install Ollama from https://ollama.ai/
ollama serve

# Pull required models
ollama pull gemma2:latest
ollama pull aroxima/gte-qwen2-1.5b-instruct:latest
```

### 4. Prepare Your Papers
```
# Create papers directory
mkdir papers

# Add your PDF files to the papers/ folder
cp your_papers/*.pdf papers/
```

### 5. Configure Fields (Optional)
Edit `fields_config.yaml` to define what information to extract:

```
- field_name: "Authors"
  description: "List of all authors, exactly as written on the first page."
- field_name: "Country"
  description: "The country or countries where the research was conducted."
```

### 6. Run the System
```
# Process all papers in the papers/ folder
python main.py

# View results
open data/results.xlsx
```

## 📁 Project Structure

```
agentic_systematic_review/
├── papers/              # Put your PDF files here
├── fields_config.yaml   # Configure what data to extract
├── data/
│   └── results.xlsx    # Output Excel file
├── chroma_db/          # Vector database storage
├── logs/               # Application logs
└── main.py             # Run this to start processing
```

## ⚙️ Advanced Usage

```
# Process specific papers only
python main.py --papers paper1.pdf paper2.pdf

# Use custom papers folder
python main.py --papers-folder /path/to/pdfs

# Force reprocess existing papers
python main.py --force

# Dry run without saving results
python main.py --dry-run

# Clear all cached embeddings
python main.py --clear-cache
```

## 🔧 Configuration

Key settings in `.env`:
- `MODEL_NAME`: Ollama model for text extraction
- `EMBEDDING_MODEL`: Model for creating vector embeddings
- `PAPERS_FOLDER`: Where to find PDF files
- `TOP_K_CHUNKS`: How many text chunks to analyze per field

## 📊 Output

Results are saved to `data/results.xlsx` with:
- One row per processed paper
- Columns for each configured field
- Metadata (confidence scores, processing time)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

[Add your license here]

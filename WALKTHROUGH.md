# Run RAGAnything with NVIDIA API

This guide explains how to run the RAGAnything system using the NVIDIA API (via OpenAI compatibility layer) and how to process documents like your Excel file.

## Why "OpenAI" in the code?

You might notice imports like `from lightrag.llm.openai import ...`.
> [!NOTE]
> Even though we are using **NVIDIA's API**, we use the **OpenAI client library** to connect to it. NVIDIA's API is designed to be "OpenAI Compatible", meaning it speaks the same language as OpenAI's servers. This is a standard industry practice that avoids rewriting code when switching providers.

## System Architecture (How it Works)

RAG-Anything uses a multi-stage pipeline to handle diverse content:

1.  **Document Parsing (via MinerU/Docling)**:
    -   Intelligently segments text, tables, and images.
    -   Preserves the original document hierarchy.

2.  **Multimodal Analysis**:
    -   **Vision**: Generates captions for images using Vision Language Models (VLM).
    -   **Tables**: Interprets structured data for better querying.
    -   **Knowledge Graph**: Builds a semantic graph of entities (e.g., "Student", "Course") and their relationships.

3.  **Intelligent Retrieval**:
    -   Combines **Vector Search** (keyword/semantic similarity) with **Graph Traversal** (following relationships).
    -   This allows answering complex questions like "List all students" by understanding the *structure* of the data, not just keyword matching.

## Setup

1.  **Environment Variables**:
    Ensure you have a `.env` file in the project root with your NVIDIA keys:
    ```bash
    LLM_BINDING_API_KEY=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
    LLM_BINDING_HOST=https://integrate.api.nvidia.com/v1
    ```
    (This file has been created for you).

2.  **Dependencies**:
    Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Code

We have updated the script to accept the file path as an argument. You no longer need to edit the code to change files.

### Basic Usage
The script now defaults to `/Users/mihirmodi/Desktop/Student List.xlsx` if no argument is provided.
```bash
python run_rag.py
```

### To Process a Different File
You can still provide a specific file path:
```bash
python run_rag.py "path/to/another/file.pdf"
```

### What Happens Next?
1.  **Processing**: The script will read your file and process it using the settings defined in `run_rag.py`.
2.  **Storage**: It creates/updates a local knowledge base in `./rag_storage`.
3.  **Query**: It runs the predefined query (currently asking to extract the student list) and prints the result.
4.  **Re-runs**: If you run the script again on the same file, it will **skip** the expensive processing (OCR, Video Analysis) and load the existing database instantly. This allows for fast, efficient querying.

## Troubleshooting

### Video & Audio Requirements
Using video/audio features requires `ffmpeg` to be installed on your system.
- **Mac**: `brew install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
- **Linux**: `sudo apt install ffmpeg`

### Processing Images and Videos

### True Multimodal Support (Images & Videos)

**Images:**
Supports JPG, PNG, etc. natively using **NVIDIA NEVA-22B** (a powerful Vision Language Model).
- Run: `python run_rag.py "photo.jpg"`
- The system will "see" the image and describe it for you.

**Videos (New!):**
Supports `.mp4`, `.avi`, `.mov`.
### Batch Processing (Folders)
You can process an entire folder of documents at once!
```bash
python run_rag.py "path/to/my_folder/"
```
The script will detect the folder and automatically process all supported files (PDF, DOCX, XLSX, Images) inside it recursively.

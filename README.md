# NLP RAG System

A Retrieval Augmented Generation (RAG) system that uses `sentence_transformers` for embeddings and `BM25` for searching.

## Technologies

- Python 3.11+
- Sentence Transformers (paraphrase-multilingual-MiniLM-L12-v2)
- Rank BM25
- Gradio (for UI)

## How to run

1. **Clone the repository:**

   ```bash
   git clone <link-to-github>
   cd <repository-folder-name>
   ```

2. **Create and activate a virtual environment:**
   Windows

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   Linux/MacOS

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables: Create a .env file in the root directory and add your configuration (e.g., API keys). You can use .env.example as a reference:**

   # Example .env content

   ```bash
   OPENAI_API_KEY=your_api_key_here
   ```

5. **Run the application:**
   ```bash
   python app.py
   ```

The application should be available at http://127.0.0.1:7860.

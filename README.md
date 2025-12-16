# NLP RAG System

A Retrieval Augmented Generation (RAG) system developed for the NLP Coursework. This project answers questions based on a custom knowledge base (Wikipedia articles) using advanced retrieval techniques including Hybrid Search and Reranking.

## Project Requirements Checklist

This project meets **100% of the course requirements** and includes the **Bonus Task**.

| Requirement | Status | Implementation Details |
| :--- | :---: | :--- |
| **Data Source** | ✅ | Automated scraping of Wikipedia (NLP, BERT, Transformers). |
| **Chunking** | ✅ | Text splitting using `RecursiveCharacterTextSplitter`. |
| **Retrieval** | ✅ | **Hybrid Search**: BM25 (Keyword) + Semantic Search (Embeddings). |
| **LLM Integration** | ✅ | Uses `LiteLLM` to support **Groq** (Llama-3) and **OpenAI**. |
| **Reranker** | ✅ | Post-retrieval re-ranking using a **Cross-Encoder**. |
| **User Interface** | ✅ | **Gradio** interface with secure API Key input. |
| **Citations (BONUS)** | 🌟 | **Implemented!** Inline citations `[ID]` and source list with scores. |

## Technologies

- Python 3.11+
- Sentence Transformers (paraphrase-multilingual-MiniLM-L12-v2)
- Cross-Encoder (ms-marco-MiniLM-L-6-v2)
- Rank BM25
- LiteLLM (for API handling)
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

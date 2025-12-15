import os
import json
import wikipedia
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = "data"
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")

CONFIGS = {
    "en": ["Natural language processing", "Large language model", "Transformer (deep learning)", "BERT (language model)", "Retrieval-augmented generation"],
    "uk": ["Обробка природної мови", "Велика мовна модель", "Трансформер (архітектура глибокого навчання)", "Штучний інтелект"]
}

def load_and_chunk():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    full_text = ""
    
    for lang, topics in CONFIGS.items():
        wikipedia.set_lang(lang)
        for topic in topics:
            try:
                page = wikipedia.page(topic, auto_suggest=False)
                print(f"[{lang.upper()}] {page.title}")
                full_text += f"\n\n=== [{lang.upper()}] {page.title} ===\n\n{page.content}"
            except Exception as e:
                print(f"Problem with '{topic}': {e}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(full_text)
    
    chunk_data = [{"id": i, "text": c} for i, c in enumerate(chunks)]
    
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, ensure_ascii=False, indent=2)
        
    print(f"Saved {len(chunks)} chunks to '{CHUNKS_FILE}'")

if __name__ == "__main__":
    load_and_chunk()
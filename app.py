import os
import gradio as gr
from dotenv import load_dotenv
from src.retriever import RAGRetriever, RAGSystem
from src.generator import RAGGenerator

load_dotenv()

rag_system = None
generator = None

def initialize_system():
    global rag_system, generator
    if os.path.exists("data/chunks.json"):
        retriever = RAGRetriever("data/chunks.json")
        rag_system = RAGSystem(retriever)
        generator = RAGGenerator()
        return "Система готова!"
    return "Файл chunks.json не знайдено. Запустіть load_data.py!"

def predict(message, history, api_key, method):
    if not rag_system:
        return "Система завантажується..."
    
    if not api_key:
        return "Помилка: Введіть API Key у меню зліва!"

    strategy_map = {
        "Гібридний (Hybrid)": "hybrid",
        "Тільки слова (BM25)": "bm25",
        "Тільки зміст (Semantic)": "semantic"
    }
    selected_strategy = strategy_map.get(method, "hybrid")
    
    docs = rag_system.retrieve(message, strategy=selected_strategy)
    
    if not docs:
        return "На жаль, інформації не знайдено."

    answer = generator.generate(message, docs, api_key=api_key)
    
    sources_text = "\n\n---\n**📚 Використані джерела:**\n"
    for d in docs:
        snippet = d['text'][:100].replace("\n", " ")
        sources_text += f"- **[{d['id']}]** (Score: {d['rerank_score']:.2f}): _{snippet}..._\n"
    
    return answer + sources_text

print(initialize_system())

with gr.Blocks(title="RAG Coursework") as demo:
    gr.Markdown("# RAG System: Hybrid Search & Reranking")
    
    with gr.Row():
        with gr.Column(scale=1):
            api_input = gr.Textbox(
                label="API Key (Groq/OpenAI)", 
                type="password",
                placeholder="Вставте ключ gsk_... тут"
            )
            method_input = gr.Radio(
                choices=["Гібридний (Hybrid)", "Тільки слова (BM25)", "Тільки зміст (Semantic)"], 
                value="Гібридний (Hybrid)", 
                label="Метод пошуку"
            )
            gr.Markdown("ℹ**BM25** - пошук за словами.\nℹ **Semantic** - пошук за змістом.\nℹ **Hybrid** - найкращий результат + Reranker.")
            
        with gr.Column(scale=4):
            gr.ChatInterface(
                fn=predict,
                additional_inputs=[api_input, method_input]
            )

if __name__ == "__main__":
    demo.launch()
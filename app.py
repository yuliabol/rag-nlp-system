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
        return "Ready!"
    return "File chunks.json not found."

def predict(message, history):
    if not rag_system:
        return "Система ініціалізується, зачекайте..."
    
    docs = rag_system.retrieve(message)
    
    if not docs:
        return "На жаль, я не знайшов інформації у базі знань."

    answer = generator.generate(message, docs)
    
    sources_text = "\n\n**Використані джерела:**\n"
    for d in docs:
        snippet = d['text'][:100].replace("\n", " ")
        sources_text += f"- **[{d['id']}]** (Score: {d['rerank_score']:.2f}): _{snippet}..._\n"
    
    return answer + sources_text

msg = initialize_system()
print(msg)

demo = gr.ChatInterface(
    fn=predict,
    title="NLP RAG System",
    description="Задайте питання про NLP, BERT, Transformers або LLM. Система знайде відповідь у Вікіпедії.",
    examples=["What is BERT?", "Хто придумав трансформери?", "Як працює RAG?"],
    #type="messages" 
)

if __name__ == "__main__":
    demo.launch()
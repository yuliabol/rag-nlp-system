import os
from litellm import completion

class RAGGenerator:
    def __init__(self, model_name="groq/llama-3.1-8b-instant"):
        self.model_name = model_name

    def generate(self, query, context_docs, api_key=None):
        if api_key:
            if "groq" in self.model_name:
                os.environ["GROQ_API_KEY"] = api_key
            elif "gpt" in self.model_name:
                os.environ["OPENAI_API_KEY"] = api_key

        context_str = ""
        for doc in context_docs:
            context_str += f"[Source {doc['id']}]: {doc['text']}\n\n"

        system_prompt = (
            "You are a helpful assistant. Use the provided Context to answer the Question. "
            "If the answer is not in the context, say you don't know. "
            "IMPORTANT: Cite your sources using [ID] notation at the end of sentences. "
            "Example: 'Transformers use attention mechanisms [12].' "
            "Answer in Ukrainian."
        )

        user_message = f"Context:\n{context_str}\n\nQuestion: {query}"

        try:
            response = completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
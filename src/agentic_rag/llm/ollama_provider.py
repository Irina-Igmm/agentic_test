from langchain_community.llms import Ollama
from agentic_rag.config import OLLAMA_MODEL

class OllamaProvider:
    @staticmethod
    def generate_response(prompt: str) -> str:
        """
        Generate a response using the Ollama model.

        Args:
            prompt: The input prompt for the model.

        Returns:
            str: The generated response.
        """
        try:
            # Initialize the Ollama client
            client = Ollama()
            response = client.complete(model=OLLAMA_MODEL, prompt=prompt)
            return response["text"].strip()
        except Exception as e:
            print(f"Error generating response with Ollama: {e}")
            return "An error occurred while generating the response."
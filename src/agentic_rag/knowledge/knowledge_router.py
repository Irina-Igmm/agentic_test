"""
Knowledge router module to determine if a query can be answered from local knowledge.
"""

from ..llm.llm_provider import llm


class KnowledgeRouter:
    """
    Router that determines if a query can be answered from local knowledge.
    """

    @staticmethod
    def check_local_knowledge(query: str, context: str) -> bool:
        prompt = """Role: Question-Answering Assistant
        Task: Determine whether the system can answer the user's question based on the provided text.
        Instructions:
            - Analyze the text and identify if it contains the necessary information to answer the user's question.
            - Provide a clear and concise response indicating whether the system can answer the question or not.
            - Your response should include only a single word. Nothing else, no other text, information, header/footer. 
        Output Format:
            - Answer: Yes/No
        Study the below examples and based on that, respond to the last question. 
        Examples:
            Input: 
                Text: The capital of France is Paris.
                User Question: What is the capital of France?
            Expected Output:
                Answer: Yes
            Input: 
                Text: The population of the United States is over 330 million.
                User Question: What is the population of China?
            Expected Output:
                Answer: No
            Input:
                User Question: {query}
                Text: {text}
        """
        formatted_prompt = prompt.format(text=context, query=query)
        print(f"Formatted Prompt: {formatted_prompt}")
        try:
            response = llm.invoke(formatted_prompt)
            cleaned_response = response.content.strip().lower()
            print(f"Model Response: {cleaned_response}")
            return cleaned_response == "yes"
        except Exception as e:
            print(f"Error invoking LLM: {e}")
            return False

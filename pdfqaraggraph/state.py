from typing import List, TypedDict

class GraphState(TypedDict):
    """
    Represents the state of the graph

    Attributes:
        question : question
        generation : LLM Generation
        documents: List of documents
    """

    question: str
    generation: str
    documents: List[str]

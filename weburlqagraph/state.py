from typing import List, TypedDict

from langchain_core.outputs import generation


class GraphState(TypedDict):
    """
    Represents the stats of our graph

    Attributes:
        question: question
        generation : LLM Generation
        web_search : whether to add search
        documents : list of documents
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]
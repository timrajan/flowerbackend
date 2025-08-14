from typing import Any, Dict

from chatgraph import graph
from weburlqagraph.chains.retrieval_grader import retrieval_grader

from weburlqagraph.state import GraphState

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag  to run the web search

    Args:
        state(dict): The current graph state
    Returns:
        state(dict): Filtered out irrelevant documents and updated web_search state

    """

    print("---CHECK THE DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_documents = []
    web_search = False

    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score

        if grade.lower()=="yes":
            print("---GRADE: DOCUMENT IS RELEVANT---")
            filtered_documents.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True
            continue

    return {"documents": filtered_documents, "question": question, "web_search": web_search}



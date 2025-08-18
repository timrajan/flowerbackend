from typing import Any, Dict
from pdfqaraggraph.chains.generation import  generation_chain
from pdfqaraggraph.state import GraphState

def generate(state: GraphState) -> Dict[str, Any]:
    print("----GENERATE----")
    question = state["question"]
    documents = state["documents"]

    # Combine documents into context
    context = "\n\n".join([doc.page_content for doc in documents])

    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}
from typing import Any, Dict
from weburlqagraph.state import GraphState
from weburlqagraph.ingestion import retriever

def retrieve(state: GraphState) -> Dict[str, Any]:
    print('Retrieving documents...')
    question=state['question']

    documents = retriever.invoke(question)
    return {"documents": documents,"question": question}


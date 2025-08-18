from typing import Any, Dict
from pdfqaraggraph.state import GraphState
from pdfqaraggraph.ingestion import retriever

def retrieve(state: GraphState) -> Dict[str, Any]:
    print('Retrieving documents...')
    question=state['question']
    print("----------------")
    print(question)
    print("----------------")
    documents = retriever.get_relevant_documents(question)
    print(documents)
    for i, doc in enumerate(documents, 1):
        content = doc.page_content
        print(f"\n{i}. {content}")  # Full content
        # OR for preview only:
        # print(f"\n{i}. {content[:500]}...")  # First 500 characters

    return {"documents": documents,"question": question}

if __name__ == '__main__':
    gs= GraphState(
        question="Few lines about daisy flower ?",
        generation="",  # Empty string for now
        documents=[]
    )
    retrieve(gs)
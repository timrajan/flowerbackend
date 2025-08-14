from typing import Any, Dict

from langchain.schema import Document
from langchain_tavily import TavilySearch

from weburlqagraph.state import  GraphState
from dotenv import load_dotenv
load_dotenv()
web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("--WEB SEARCH--")
    question = state["question"]
    documents = state["documents"]

    tavily_results = web_search_tool.invoke({"query": question})




    joined_tavily_results = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results["results"]]
    )

    web_results = Document(page_content=joined_tavily_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]

    return {"documents": documents, "question": question}


if __name__ == "__main__":
    web_search(state={"question": "what is langchain ?", "documents": None})

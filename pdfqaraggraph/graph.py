from dotenv import load_dotenv

from langgraph.graph import END,StateGraph,START

from pdfqaraggraph.consts import RETRIEVE,GENERATE

from pdfqaraggraph.nodes.retrieve import retrieve
from pdfqaraggraph.nodes.generate import generate

from pdfqaraggraph.state import GraphState


load_dotenv()

workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE,retrieve)

workflow.add_node(GENERATE,generate)
workflow.add_edge(START, RETRIEVE)
workflow.add_edge(RETRIEVE,GENERATE)
workflow.add_edge(GENERATE,END)

app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="pdfgraph.png")
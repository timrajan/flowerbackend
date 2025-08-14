from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import asyncio
import uvicorn
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables from all .env files
import pathlib


def load_all_env_files():
    """Load environment variables from all graph folders"""
    project_root = pathlib.Path(__file__).parent

    # Load from root .env if it exists
    root_env = project_root / ".env"
    if root_env.exists():
        print(f"Loading environment from: {root_env}")
        load_dotenv(root_env)

    # Graph folders that might have .env files
    graph_folders = [
        "chatgraph",
        "emailgraph",
        "messagegraph",
        "pdfqaraggraph",
        "voicegraph",
        "weburlqagraph",
        "whatsappgraph"
    ]

    # Load .env from each graph folder
    for folder in graph_folders:
        env_file = project_root / folder / ".env"
        if env_file.exists():
            print(f"Loading environment from: {env_file}")
            load_dotenv(env_file, override=False)  # Don't override already set vars


# Load all environment variables
load_all_env_files()


# Import your graph apps from main.py files
def import_graph_app(module_name):
    """Helper function to import graph apps from main.py files"""
    print(f"🔄 Attempting to import {module_name}...")
    try:
        # Import the app from module.graph (like your weburlqagraph.graph import app)
        print(f"  Trying: from {module_name}.graph import app")
        module = __import__(f"{module_name}.graph", fromlist=["app"])

        if hasattr(module, "app"):
            app_obj = getattr(module, "app")
            print(f"  ✅ Found app object: {type(app_obj)}")

            # Check if it has invoke method
            if hasattr(app_obj, 'invoke'):
                print(f"  ✅ App has invoke method")
                return app_obj
            else:
                print(
                    f"  ❌ App missing invoke method. Available methods: {[m for m in dir(app_obj) if not m.startswith('_')]}")
                return None
        else:
            print(f"  ❌ No 'app' attribute found in {module_name}.graph")
            available_attrs = [attr for attr in dir(module) if not attr.startswith('_')]
            print(f"  Available attributes: {available_attrs}")
            return None

    except ImportError as e:
        print(f"  ❌ ImportError: {e}")
        return None
    except Exception as e:
        print(f"  ❌ Other error: {e}")
        return None


print("🚀 Starting graph imports...")

# Try to import all graph apps
graphs = {}

graph_modules = {
    "chat": "chatgraph",
    "email": "emailgraph",
    "message": "messagegraph",
    "pdfqara": "pdfqaraggraph",
    "voice": "voicegraph",
    "weburl": "weburlqagraph",  # This will import app from weburlqagraph.graph
    "whatsapp": "whatsappgraph"
}

for graph_name, module_name in graph_modules.items():
    app_obj = import_graph_app(module_name)
    if app_obj is not None:
        graphs[graph_name] = app_obj
        print(f"✅ Successfully imported {graph_name} app from {module_name}")
    else:
        print(f"❌ Failed to import {graph_name} app from {module_name}")

print(f"📊 Import summary: {len(graphs)}/{len(graph_modules)} graphs loaded")
print(f"Successfully loaded: {list(graphs.keys())}")

# Available graphs mapping
AVAILABLE_GRAPHS = graphs


# Request/Response models
class GraphRequest(BaseModel):
    input_data: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None


class GraphResponse(BaseModel):
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting FlowerBackend API server...")
    print(f"Available graphs: {list(AVAILABLE_GRAPHS.keys())}")
    yield
    # Shutdown
    print("Shutting down FlowerBackend API server...")


app = FastAPI(
    title="FlowerBackend API",
    description="API for invoking LangGraph workflows",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    return {
        "message": "FlowerBackend API is running!",
        "available_graphs": list(AVAILABLE_GRAPHS.keys()),
        "endpoints": {
            "invoke": "/invoke/{graph_name}",
            "stream": "/stream/{graph_name}",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "flowerbackend"}


@app.get("/debug/{graph_name}")
async def debug_graph(graph_name: str):
    """
    Debug endpoint to inspect a graph's available methods and attributes
    """
    if graph_name not in AVAILABLE_GRAPHS:
        raise HTTPException(
            status_code=404,
            detail=f"Graph '{graph_name}' not found. Available graphs: {list(AVAILABLE_GRAPHS.keys())}"
        )

    try:
        graph_app = AVAILABLE_GRAPHS[graph_name]

        # Get all public methods and attributes
        methods = [attr for attr in dir(graph_app) if not attr.startswith('_')]

        # Check specifically for invoke methods
        invoke_methods = [attr for attr in methods if 'invoke' in attr.lower()]

        # Get graph type
        graph_type = str(type(graph_app))

        return {
            "graph_name": graph_name,
            "graph_type": graph_type,
            "all_methods": methods,
            "invoke_methods": invoke_methods,
            "has_invoke": hasattr(graph_app, 'invoke'),
            "has_stream": hasattr(graph_app, 'stream')
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error debugging graph: {str(e)}")


@app.get("/graphs")
async def list_graphs():
    """List all available graphs with their debug info"""
    graphs_info = {}

    for graph_name in AVAILABLE_GRAPHS:
        try:
            graph_app = AVAILABLE_GRAPHS[graph_name]
            graphs_info[graph_name] = {
                "type": str(type(graph_app)),
                "has_invoke": hasattr(graph_app, 'invoke'),
                "has_stream": hasattr(graph_app, 'stream'),
                "status": "loaded"
            }
        except Exception as e:
            graphs_info[graph_name] = {
                "status": "error",
                "error": str(e)
            }

    return {
        "available_graphs": list(AVAILABLE_GRAPHS.keys()),
        "total_count": len(AVAILABLE_GRAPHS),
        "graphs_info": graphs_info
    }


@app.post("/invoke/{graph_name}")
async def invoke_graph(graph_name: str, request: GraphRequest) -> GraphResponse:
    """
    Invoke a specific graph with the provided input data.
    Uses the synchronous invoke method in a thread pool to avoid blocking.
    """
    if graph_name not in AVAILABLE_GRAPHS:
        raise HTTPException(
            status_code=404,
            detail=f"Graph '{graph_name}' not found. Available graphs: {list(AVAILABLE_GRAPHS.keys())}"
        )

    try:
        graph_app = AVAILABLE_GRAPHS[graph_name]

        if not hasattr(graph_app, 'invoke'):
            return GraphResponse(
                success=False,
                error=f"Graph app '{graph_name}' has no 'invoke' method. Available methods: {[attr for attr in dir(graph_app) if not attr.startswith('_')]}"
            )

        # Run the synchronous invoke in a thread pool to avoid blocking FastAPI
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: graph_app.invoke(input=request.input_data, config=request.config or {})
        )

        return GraphResponse(success=True, result=result)

    except Exception as e:
        return GraphResponse(
            success=False,
            error=f"Error invoking graph '{graph_name}': {str(e)}"
        )


@app.post("/stream/{graph_name}")
async def stream_graph(graph_name: str, request: GraphRequest):
    """
    Stream results from a specific graph.
    Uses synchronous streaming in a thread pool if available.
    """
    if graph_name not in AVAILABLE_GRAPHS:
        raise HTTPException(
            status_code=404,
            detail=f"Graph '{graph_name}' not found. Available graphs: {list(AVAILABLE_GRAPHS.keys())}"
        )

    try:
        from fastapi.responses import StreamingResponse
        import json

        graph_app = AVAILABLE_GRAPHS[graph_name]

        async def generate_stream():
            try:
                if hasattr(graph_app, 'stream'):
                    # Use synchronous stream method
                    for chunk in graph_app.stream(
                            input=request.input_data,
                            config=request.config or {}
                    ):
                        yield f"data: {json.dumps(chunk)}\n\n"
                else:
                    yield f"data: {json.dumps({'error': 'No streaming method available'})}\n\n"

                yield f"data: {json.dumps({'_end': True})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Specific endpoints for each graph type (optional convenience endpoints)
@app.post("/chat")
async def chat_endpoint(request: GraphRequest) -> GraphResponse:
    """Convenience endpoint for chat graph"""
    return await invoke_graph("chat", request)


@app.post("/email")
async def email_endpoint(request: GraphRequest) -> GraphResponse:
    """Convenience endpoint for email graph"""
    return await invoke_graph("email", request)


@app.post("/message")
async def message_endpoint(request: GraphRequest) -> GraphResponse:
    """Convenience endpoint for message graph"""
    return await invoke_graph("message", request)


@app.post("/pdfqara")
async def pdfqara_endpoint(request: GraphRequest) -> GraphResponse:
    """Convenience endpoint for PDF QA graph"""
    return await invoke_graph("pdfqara", request)


@app.post("/voice")
async def voice_endpoint(request: GraphRequest) -> GraphResponse:
    """Convenience endpoint for voice graph"""
    return await invoke_graph("voice", request)


@app.post("/weburl")
async def weburl_endpoint(request: GraphRequest) -> GraphResponse:
    """Convenience endpoint for web URL QA graph"""
    return await invoke_graph("weburl", request)


@app.post("/whatsapp")
async def whatsapp_endpoint(request: GraphRequest) -> GraphResponse:
    """Convenience endpoint for WhatsApp graph"""
    return await invoke_graph("whatsapp", request)


if __name__ == "__main__":
    # Get port from environment variable (Railway sets this)
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    print(f"🚀 Starting server on {host}:{port}")

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=False,  # Disable reload in production
        log_level="info"
    )
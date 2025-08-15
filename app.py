import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
import asyncio

print("🚀 Starting FlowerBackend API with Lazy Loading...")
print(f"🔍 OPENAI_API_KEY present: {bool(os.getenv('OPENAI_API_KEY'))}")


# Lazy wrapper class
class LazyGraphWrapper:
    """Wrapper that delays graph initialization until first use"""

    def __init__(self, module_name: str):
        self.module_name = module_name
        self._app = None
        self._initialized = False
        self._error = None

    def _ensure_initialized(self):
        """Initialize the graph app only when needed"""
        if self._initialized:
            return

        print(f"🔄 Lazy loading {self.module_name}...")

        # Ensure environment variables are set
        if not os.getenv("OPENAI_API_KEY"):
            self._error = f"OPENAI_API_KEY not found when initializing {self.module_name}"
            raise ValueError(self._error)

        try:
            # Import the graph module
            module = __import__(f"{self.module_name}.graph", fromlist=["app"])

            if hasattr(module, "app"):
                self._app = module.app
                print(f"✅ Successfully lazy-loaded {self.module_name}")
            else:
                self._error = f"No 'app' attribute found in {self.module_name}.graph"
                raise AttributeError(self._error)

        except Exception as e:
            self._error = str(e)
            print(f"❌ Failed to lazy-load {self.module_name}: {e}")
            raise

        self._initialized = True

    def invoke(self, input: Dict[str, Any], config: Dict[str, Any] = None):
        """Invoke the wrapped graph app"""
        self._ensure_initialized()
        return self._app.invoke(input=input, config=config or {})

    @property
    def is_available(self):
        """Check if the graph can be loaded"""
        if self._error:
            return False
        try:
            self._ensure_initialized()
            return True
        except Exception as e:
            self._error = str(e)
            return False

    @property
    def error_message(self):
        """Get the error message if graph failed to load"""
        return self._error


# Create lazy wrappers for all graphs
print("🔄 Creating lazy graph wrappers...")
AVAILABLE_GRAPHS = {
    "weburl": LazyGraphWrapper("weburlqagraph"),
    "chat": LazyGraphWrapper("chatgraph"),
    "email": LazyGraphWrapper("emailgraph"),
    "message": LazyGraphWrapper("messagegraph"),
    "pdfqara": LazyGraphWrapper("pdfqaraggraph"),
    "voice": LazyGraphWrapper("voicegraph"),
    "whatsapp": LazyGraphWrapper("whatsappgraph")
}

print(f"✅ Created {len(AVAILABLE_GRAPHS)} lazy graph wrappers")

# Create FastAPI app
app = FastAPI(title="FlowerBackend API", description="LangGraph FastAPI Backend")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",      # Local development
        "https://aiflowershop.com",   # Production frontend
        # Add any other frontend origins you need
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


class GraphRequest(BaseModel):
    input_data: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None


class GraphResponse(BaseModel):
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@app.get("/")
async def root():
    return {
        "message": "FlowerBackend API with Lazy Loading",
        "status": "running",
        "total_graphs": len(AVAILABLE_GRAPHS),
        "available_endpoints": {
            "health": "/health",
            "graphs": "/graphs",
            "invoke": "/invoke/{graph_name}",
            "debug": "/debug"
        },
        "openai_key_present": bool(os.getenv("OPENAI_API_KEY")),
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "flowerbackend"}


@app.get("/debug")
async def debug_info():
    """Debug endpoint for troubleshooting"""
    return {
        "environment_variables": {
            key: "***SET***" if value else "***NOT SET***"
            for key, value in os.environ.items()
            if any(keyword in key.upper() for keyword in ['API', 'KEY', 'OPENAI', 'LANG'])
        },
        "graphs_status": {
            name: {
                "available": wrapper.is_available,
                "error": wrapper.error_message,
                "initialized": wrapper._initialized
            }
            for name, wrapper in AVAILABLE_GRAPHS.items()
        }
    }


@app.get("/graphs")
async def list_graphs():
    """List available graphs and test their availability"""
    graphs_info = {}
    available_graphs = []

    for graph_name, wrapper in AVAILABLE_GRAPHS.items():
        is_available = wrapper.is_available
        graphs_info[graph_name] = {
            "status": "available" if is_available else "error",
            "loaded": wrapper._initialized,
            "error": wrapper.error_message if not is_available else None
        }
        if is_available:
            available_graphs.append(graph_name)

    return {
        "available_graphs": available_graphs,
        "total_count": len(available_graphs),
        "graphs_info": graphs_info
    }


@app.post("/invoke/{graph_name}")
async def invoke_graph(graph_name: str, request: GraphRequest):
    """Invoke a specific graph"""
    if graph_name not in AVAILABLE_GRAPHS:
        raise HTTPException(
            status_code=404,
            detail=f"Graph '{graph_name}' not found. Available: {list(AVAILABLE_GRAPHS.keys())}"
        )

    wrapper = AVAILABLE_GRAPHS[graph_name]

    # Check if graph is available before trying to invoke
    if not wrapper.is_available:
        raise HTTPException(
            status_code=500,
            detail=f"Graph '{graph_name}' is not available. Error: {wrapper.error_message}"
        )

    try:
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: wrapper.invoke(input=request.input_data, config=request.config or {})
        )

        return GraphResponse(success=True, result=result)

    except Exception as e:
        return GraphResponse(
            success=False,
            error=f"Error invoking graph '{graph_name}': {str(e)}"
        )


# Convenience endpoints for each graph
@app.post("/weburl")
async def weburl_endpoint(request: GraphRequest):
    """Convenience endpoint for weburl graph"""
    return await invoke_graph("weburl", request)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"

    print(f"🚀 Starting server on {host}:{port}")
    uvicorn.run("app:app", host=host, port=port, reload=False, log_level="info")
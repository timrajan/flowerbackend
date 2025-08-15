"""
Wrapper for graph modules to handle lazy initialization
"""
import os
from typing import Any, Dict


class LazyGraphWrapper:
    """Wrapper that delays graph initialization until first use"""

    def __init__(self, module_name: str):
        self.module_name = module_name
        self._app = None
        self._initialized = False

    def _ensure_initialized(self):
        """Initialize the graph app only when needed"""
        if self._initialized:
            return

        print(f"🔄 Lazy loading {self.module_name}...")

        # Ensure environment variables are set
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(f"OPENAI_API_KEY not found when initializing {self.module_name}")

        try:
            # Import the graph module
            module = __import__(f"{self.module_name}.graph", fromlist=["app"])

            if hasattr(module, "app"):
                self._app = module.app
                print(f"✅ Successfully lazy-loaded {self.module_name}")
            else:
                raise AttributeError(f"No 'app' attribute found in {self.module_name}.graph")

        except Exception as e:
            print(f"❌ Failed to lazy-load {self.module_name}: {e}")
            raise

        self._initialized = True

    def invoke(self, input: Dict[str, Any], config: Dict[str, Any] = None):
        """Invoke the wrapped graph app"""
        self._ensure_initialized()
        return self._app.invoke(input=input, config=config or {})

    def stream(self, input: Dict[str, Any], config: Dict[str, Any] = None):
        """Stream from the wrapped graph app"""
        self._ensure_initialized()
        if hasattr(self._app, 'stream'):
            return self._app.stream(input=input, config=config or {})
        else:
            raise AttributeError(f"Graph app in {self.module_name} has no 'stream' method")

    def __getattr__(self, name):
        """Delegate other method calls to the wrapped app"""
        self._ensure_initialized()
        return getattr(self._app, name)
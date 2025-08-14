#!/usr/bin/env python3
"""
Debug script to test imports and see what's available
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"✅ Loaded .env from: {env_file}")
else:
    print(f"❌ No .env file found at: {env_file}")


def test_single_import():
    """Test importing just the weburlqagraph app"""
    print("\n🔍 Testing weburlqagraph import...")

    try:
        print("Step 1: Trying to import weburlqagraph module...")
        import weburlqagraph
        print(f"✅ weburlqagraph module imported: {weburlqagraph}")

        print("Step 2: Trying to import weburlqagraph.graph...")
        from weburlqagraph import graph as graph_module
        print(f"✅ weburlqagraph.graph imported: {graph_module}")

        print("Step 3: Checking what's in the graph module...")
        graph_attrs = [attr for attr in dir(graph_module) if not attr.startswith('_')]
        print(f"Available attributes: {graph_attrs}")

        print("Step 4: Trying to get 'app' from graph module...")
        if hasattr(graph_module, 'app'):
            app = graph_module.app
            print(f"✅ Found app: {app}")
            print(f"App type: {type(app)}")

            app_methods = [attr for attr in dir(app) if not attr.startswith('_')]
            print(f"App methods: {app_methods}")

            if hasattr(app, 'invoke'):
                print("✅ App has 'invoke' method")

                # Test invoke
                print("Step 5: Testing invoke...")
                result = app.invoke(input={"question": "test"})
                print(f"✅ Invoke successful: {type(result)}")
                return app
            else:
                print("❌ App doesn't have 'invoke' method")
        else:
            print("❌ No 'app' attribute found in graph module")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    return None


def test_all_imports():
    """Test importing all graph modules"""
    print("\n🔍 Testing all graph imports...")

    graph_modules = {
        "chat": "chatgraph",
        "email": "emailgraph",
        "message": "messagegraph",
        "pdfqara": "pdfqaraggraph",
        "voice": "voicegraph",
        "weburl": "weburlqagraph",
        "whatsapp": "whatsappgraph"
    }

    successful_imports = {}

    for graph_name, module_name in graph_modules.items():
        print(f"\nTesting {graph_name} ({module_name})...")
        try:
            # Try to import the module
            module = __import__(module_name)
            print(f"  ✅ Module {module_name} imported")

            # Try to import the graph submodule
            graph_module = __import__(f"{module_name}.graph", fromlist=["app"])
            print(f"  ✅ {module_name}.graph imported")

            # Try to get the app
            if hasattr(graph_module, 'app'):
                app = getattr(graph_module, 'app')
                print(f"  ✅ Found app: {type(app)}")

                if hasattr(app, 'invoke'):
                    print(f"  ✅ App has invoke method")
                    successful_imports[graph_name] = app
                else:
                    print(f"  ❌ App missing invoke method")
            else:
                print(f"  ❌ No 'app' attribute in {module_name}.graph")

        except Exception as e:
            print(f"  ❌ Error importing {module_name}: {e}")

    print(f"\n📊 Summary: Successfully imported {len(successful_imports)} graphs")
    print(f"Successful graphs: {list(successful_imports.keys())}")

    return successful_imports


def check_directory_structure():
    """Check the current directory structure"""
    print("\n📁 Checking directory structure...")
    project_root = Path(__file__).parent
    print(f"Current directory: {project_root}")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries

    # Check if graph folders exist
    graph_folders = ["chatgraph", "emailgraph", "messagegraph", "pdfqaraggraph",
                     "voicegraph", "weburlqagraph", "whatsappgraph"]

    for folder in graph_folders:
        folder_path = project_root / folder
        if folder_path.exists():
            print(f"  ✅ {folder} exists")

            # Check for graph.py
            graph_py = folder_path / "graph.py"
            if graph_py.exists():
                print(f"    ✅ {folder}/graph.py exists")
            else:
                print(f"    ❌ {folder}/graph.py missing")

            # Check for __init__.py
            init_py = folder_path / "__init__.py"
            if init_py.exists():
                print(f"    ✅ {folder}/__init__.py exists")
            else:
                print(f"    ❌ {folder}/__init__.py missing")
        else:
            print(f"  ❌ {folder} doesn't exist")


if __name__ == "__main__":
    print("🚀 Starting import diagnostics...")

    check_directory_structure()
    test_single_import()
    successful_graphs = test_all_imports()

    print(f"\n🎯 Final result: {len(successful_graphs)} graphs ready for FastAPI")
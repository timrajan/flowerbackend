import os
from fastapi import FastAPI
import uvicorn

print("🚀 Starting super simple FastAPI app...")
print(f"🔍 OPENAI_API_KEY present: {bool(os.getenv('OPENAI_API_KEY'))}")

app = FastAPI(title="FlowerBackend Simple API")


@app.get("/")
def root():
    return {
        "message": "FlowerBackend Simple API is running!",
        "openai_key_present": bool(os.getenv("OPENAI_API_KEY")),
        "environment_count": len([k for k in os.environ.keys() if 'API' in k or 'KEY' in k])
    }


@app.get("/health")
def health():
    return {"status": "healthy", "service": "flowerbackend"}


@app.get("/env-debug")
def env_debug():
    """Debug endpoint to check environment variables"""
    env_vars = {}
    for key, value in os.environ.items():
        if any(keyword in key.upper() for keyword in ['API', 'KEY', 'OPENAI', 'LANG']):
            env_vars[key] = "***SET***" if value else "***NOT SET***"

    return {
        "environment_variables": env_vars,
        "total_env_vars": len(os.environ),
        "app_working": True
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"

    print(f"🚀 Starting on {host}:{port}")
    print(
        f"📋 Environment variables with 'API' or 'KEY': {len([k for k in os.environ.keys() if 'API' in k or 'KEY' in k])}")

    uvicorn.run(
        "app:app",  # This should match the filename
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
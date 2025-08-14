#!/bin/bash

# Start script for Railway deployment
echo "🚀 Starting FlowerBackend API on Railway..."

# Set environment variables for Railway
export HOST=0.0.0.0
export PORT=${PORT:-8000}

# Start the FastAPI application
python app.py
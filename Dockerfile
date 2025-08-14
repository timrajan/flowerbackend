# Use Python 3.11.4 to match your Pipfile
FROM python:3.11.4-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for your packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy Pipfile and Pipfile.lock first (for better Docker layer caching)
COPY Pipfile Pipfile.lock ./

# Install pipenv and dependencies
RUN pip install --no-cache-dir pipenv && \
    pipenv install --system --deploy

# Copy the rest of the application
COPY . .

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["python", "app.py"]
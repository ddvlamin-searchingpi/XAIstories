FROM mcr.microsoft.com/devcontainers/python:3.13

RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy requirements.txt
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if needed (e.g., for certain python libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Add the extra index URL for PyTorch CPU
RUN pip install --no-cache-dir \
    -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
# Copy the rest of the application
COPY . .

# Ensure uvicorn is called correctly
# Assuming your main app is in main.py and named 'app'
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
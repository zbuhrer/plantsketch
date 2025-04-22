FROM python:3.9

WORKDIR /app

# Install system dependencies for OpenCV and Open3D
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install dependencies directly (no virtual env needed in container)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p projects

# Volume for persistent storage
VOLUME ["/app/projects"]

# Expose Streamlit port
EXPOSE 8501

# Configure environment variables for Streamlit
ENV PYTHONPATH=/app

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

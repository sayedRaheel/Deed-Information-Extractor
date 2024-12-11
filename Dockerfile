# Use RunPod PyTorch image with CUDA support
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download and cache the OCR model
RUN python3 -c "from doctr.models import ocr_predictor; ocr_predictor(pretrained=True)"

# Copy only necessary files
COPY handler.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Healthcheck to verify GPU availability
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; assert torch.cuda.is_available(), 'GPU not available'"

# Run the handler
CMD [ "python", "-u", "handler.py" ]
# Use official lightweight Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Streamlit application, theme, and configurations
COPY app_v2.py theme.py ./
COPY .streamlit/ ./.streamlit/

# Streamlit port and address configurations
EXPOSE 8501

# Run streamlit with dynamic PORT binding
CMD ["sh", "-c", "streamlit run app_v2.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]

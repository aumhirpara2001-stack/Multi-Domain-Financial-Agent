FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application
COPY . /app

# Default command runs evaluation
CMD ["python", "scripts/evaluate.py"]

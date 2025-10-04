
FROM python:3.10-slim

WORKDIR /app

# Install basic dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python app (converted from .ipynb)
COPY app.ipynb .

EXPOSE 7860

CMD ["python", "app.ipynb"]

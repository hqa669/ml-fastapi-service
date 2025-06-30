FROM python:3.9-slim

WORKDIR /app

# Copy application and model
COPY ./api /app/api
COPY ./requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
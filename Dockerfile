# Container for the digit recognizer web app.
# Build:  docker build -t digit-recognizer .
# Run:    docker run -p 8000:8000 digit-recognizer

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bring in the app code AND the trained model.
COPY app/ ./app/
COPY model/saved_models/mnist_cnn_model.h5 ./model/saved_models/mnist_cnn_model.h5

ENV PORT=8000
EXPOSE 8000

# Most PaaS providers (Render, Railway, Fly.io) inject $PORT at runtime.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

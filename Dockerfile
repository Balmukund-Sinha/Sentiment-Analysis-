# Stage 1: Build
FROM python:3.9-slim as builder

WORKDIR /app
COPY requirements.txt .
# Install with --user to isolate
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.9-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy app code
COPY backend/ ./backend/
COPY calibration.py .

# Copy models (Assuming they are mounted or copied in CI/CD)
# For local generic build we create dir
RUN mkdir saved_model

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]

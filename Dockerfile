FROM python:3.11-slim

WORKDIR /app

# Copy code
COPY . .

# Install only what we need to run (none here, but keep pattern)
RUN python -m pip install --upgrade pip

# Run the app
CMD ["python", "-m", "src.app"]

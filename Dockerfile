# Use an official Python image
FROM python:3.10-slim

# Install system dependencies for OpenCV + libGL
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy everything
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 
EXPOSE 8000

# Run the app using gunicorn
#CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
CMD ["gunicorn", "--timeout", "120", "--bind", "0.0.0.0:8000", "app:app"]


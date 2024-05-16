# Use an official Python runtime as a parent image
FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Update and install necessary dependencies
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    ubuntu-drivers-common \
    gcc \
    nvidia-cuda-toolkit \
    python3.11 \
    python3-pip \
    screen \
    ffmpeg \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Remove the EXTERNALLY-MANAGED file
RUN rm /usr/lib/python3.*/EXTERNALLY-MANAGED

# Clone the repository
RUN git clone https://github.com/clalanliu/insanely-fast-whisper-api.git /insanely-fast-whisper-api

# Set working directory
WORKDIR /insanely-fast-whisper-api

# Install requirements from requirements.txt
COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Install Python dependencies
RUN python3.11 -m pip install --no-cache-dir -U wheel ninja packaging && \
    python3.11 -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


# Expose the port uvicorn will run on
EXPOSE 8000

# Launch the application
CMD ["python3.11", "-m", "uvicorn", "app.app:app", "--reload"]

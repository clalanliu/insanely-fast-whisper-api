# Use an official Python runtime as a parent image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Update and install necessary dependencies
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    ubuntu-drivers-common \
    gcc \
    screen \
    ffmpeg \
    curl \
    git
RUN apt-get update -y && apt-get install python3-pip -y

# Update alternatives to set python3.11 and pip as default
RUN python3 -m pip install -U wheel ninja packaging
RUN apt-get update -y && apt-get install uvicorn -y

# Clone the repository
RUN git clone https://github.com/clalanliu/insanely-fast-whisper-api.git /insanely-fast-whisper-api
# Set working directory
WORKDIR /insanely-fast-whisper-api

# Install requirements from requirements.txt
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install uvicorn fastapi transformers pyannote.audio
RUN python3 -m pip install torch==2.3.0 torchvision torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Expose the port uvicorn will run on
EXPOSE 8000

# Launch the application
CMD ["python3", "-m", "uvicorn", "app.app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]

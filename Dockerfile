FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1-mesa-glx

WORKDIR /ghost
COPY . .

RUN pip3 install --upgrade pip setuptools wheel && pip3 install -r requirements-docker.txt

ENTRYPOINT ["python", "inference.py"]
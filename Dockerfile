FROM python:3.11-slim

# Prevent SegFaults by limiting thread contention between ONNX and TensorFlow
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# Install system dependencies required by OpenCV and wget for downloading models
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libxcb-render0 \
    libxcb-xinerama0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download the heavy models directly from GitHub Releases into the models/ folder
RUN mkdir -p models && \
    wget -q -O models/face_landmarks_lite.tflite "https://github.com/izharadrali/Face_Recognition_API/releases/download/models/face_landmarks_lite.tflite" && \
    wget -q -O models/r50.onnx "https://github.com/izharadrali/Face_Recognition_API/releases/download/models/r50.onnx" && \
    wget -q -O models/yolov8n-face-lindevs.onnx "https://github.com/izharadrali/Face_Recognition_API/releases/download/models/yolov8n-face-lindevs.onnx"

EXPOSE 7000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7000"]
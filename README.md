# Face_Recognition_API
This api is designed for registering person/employ for recognition purpose on edge device .



# Face Recognition API

A high-performance Face Embedding API built with FastAPI. This service extracts robust facial embeddings from uploaded images using a multi-stage pipeline: face detection (YOLO), facial alignment (Landmarks), and embedding extraction (ArcFace). 

## Features
* **FastAPI Backend:** Provides a fast, asynchronous REST API.
* **Multi-Stage Processing Pipeline:**
  * **Detection:** Utilizes YOLOv8 (via PyTorch) for accurate face detection[cite: 2].
  * **Alignment:** Uses a TFLite landmark model to extract facial landmarks and affine-warp (align) the face crops[cite: 2].
  * **Embedding:** Uses ONNX Runtime with an ArcFace model (MobileFaceNet/MBF) to generate normalized 512-dimensional embeddings[cite: 2].
* **Multi-Image Averaging:** Accepts up to 5 images per request and returns a normalized average embedding for higher accuracy.
* **Optimization & Safety:** 
  * Automatically downscales images larger than 1024px to maintain fast processing speeds.
  * Gracefully drops invalid images or images without detected faces without crashing the API.
  * Enforces strict thread limits on ONNX, PyTorch, and TensorFlow to prevent SegFaults and CPU contention[cite: 1, 2].
* **Docker Ready:** Includes a `Dockerfile` with optimized system dependencies for OpenCV.

## API Endpoint
* **`POST /get-embedding`**[cite: 3]
  * **Form Data:** `file` (List of up to 5 images)[cite: 3].
  * **Returns:** A JSON object containing a status message, a list of failed image filenames, and the final normalized face embedding array[cite: 3].

## Tech Stack
* Python 3.11
* FastAPI & Uvicorn[cite: 3, 4]
* PyTorch / Ultralytics[cite: 2, 4]
* ONNX Runtime & TFLite Runtime[cite: 2, 4]
* OpenCV (Headless)[cite: 4]

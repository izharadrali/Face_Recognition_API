import sys
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import numpy as np
import cv2

from face_service import FaceExtractor

app = FastAPI(
    title="Face Embedding API",
    version="2.0"
)

try:
    extractor = FaceExtractor(
        yolo_path="models/yolov8n-face-lindevs.onnx",
        landmark_path="models/face_landmarks_lite.tflite",
        arcface_path="models/r50.onnx",
        face_class_id=0
    )
    print("Models loaded into RAM.")
    
    # NEW: Force a dummy run to pre-warm the memory buffers
    print("Warming up models...")
    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
    try:
        extractor.extract_embedding_from_image(dummy_image)
    except Exception:
        pass # We expect this to fail or return zeros, we just need the models to cycle once.
    print("Warm-up complete. API is ready.")
    
except Exception as e:
    print(f"FATAL ERROR: Failed to load models. {e}")
    sys.exit(1)


@app.post("/get-embedding")
async def get_embedding(file: List[UploadFile] = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    if len(file) > 5:
        raise HTTPException(
            status_code=400,
            detail="Maximum of 5 images allowed per request."
        )

    embeddings = []
    failed_files = []  # List to track images that drop out

    for uploaded_file in file:
        if not uploaded_file.filename:
            continue

        try:
            contents = await uploaded_file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                failed_files.append(f"{uploaded_file.filename} (Invalid image format)")
                continue

            # --- NEW: Fast Pre-Scaling for massive images ---
            h, w = image.shape[:2]
            max_dim = 1024
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            # ------------------------------------------------

            # Attempt to extract embedding
            emb, timings = extractor.extract_embedding_from_image(image)
            embeddings.append(emb)

        except ValueError as ve:
            # Instead of crashing the whole API, we just log the failure and move on
            failed_files.append(f"{uploaded_file.filename} ({str(ve)})")
            continue

    # If ALL uploaded images failed (e.g., 3 pictures of cars)
    if not embeddings:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "No valid faces could be extracted from any images.",
                "failed_images": failed_files
            }
        )

    # Calculate average embedding for the successful images
    if len(embeddings) == 1:
        final_embedding = embeddings[0]
    else:
        avg_embedding = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg_embedding)
        final_embedding = avg_embedding / norm if norm > 0 else avg_embedding

    # Create a dynamic status message
    if failed_files:
        status_msg = f"Successfully averaged {len(embeddings)} face(s). Dropped {len(failed_files)} image(s)."
    else:
        status_msg = f"Successfully averaged all {len(embeddings)} face(s)."

    # Return a structured JSON dictionary
    return {
        "message": status_msg,
        "failed_images": failed_files,
        "embedding": final_embedding.tolist()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7000, reload=True)
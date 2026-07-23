import cv2
import numpy as np
import onnxruntime as ort
import tflite_runtime.interpreter as tflite
import time
import torch # <-- ADDED IMPORT
from ultralytics import YOLO

class FaceExtractor:
    def __init__(self, yolo_path, arcface_path, landmark_path, face_class_id=0):
        self.arcface_input_size = (112, 112)
        self.landmark_input_size = (192, 192)
        self.face_class_id = face_class_id

        # 1. Limit PyTorch (YOLO) Threads
        torch.set_num_threads(2)

        # Same as batch code
        print("Loading YOLO...", flush=True)
        self.detector = YOLO(yolo_path, task="detect")
        print("YOLO loaded", flush=True)

        # 2. Limit ONNX Runtime (ArcFace) Threads
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        providers = ["CPUExecutionProvider"]
        self.arcface_sess = ort.InferenceSession(
            arcface_path, 
            sess_options=opts, # <-- ADDED THIS
            providers=providers
        )
        self.arcface_input_name = self.arcface_sess.get_inputs()[0].name

        # 3. Limit TFLite (Landmarks) Threads
        self.landmark_interpreter = tflite.Interpreter(
            model_path=landmark_path,
            num_threads=1 # <-- ADDED THIS
        )

        self.landmark_interpreter.allocate_tensors()
        self.land_input_details = self.landmark_interpreter.get_input_details()
        self.land_output_details = self.landmark_interpreter.get_output_details()

        self.ref_pts = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)

    def preprocess_landmarks(self, face_img):
        resized = cv2.resize(face_img, self.landmark_input_size)
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        batch = np.expand_dims(rgb_img.astype(np.float32) / 255.0, axis=0)
        return batch, resized.shape[1], resized.shape[0]

    def extract_5_points_from_468(self, landmarks_468):
        left_eye = np.mean([landmarks_468[33], landmarks_468[133]], axis=0)
        right_eye = np.mean([landmarks_468[263], landmarks_468[362]], axis=0)
        nose = landmarks_468[1]
        left_mouth = landmarks_468[61]
        right_mouth = landmarks_468[291]

        return np.array(
            [left_eye, right_eye, nose, left_mouth, right_mouth],
            dtype=np.float32
        )

    def extract_5_points_from_68(self, landmarks_68):
        left_eye = np.mean(landmarks_68[36:42], axis=0)
        right_eye = np.mean(landmarks_68[42:48], axis=0)
        nose = landmarks_68[30]
        left_mouth = landmarks_68[48]
        right_mouth = landmarks_68[54]

        return np.array(
            [left_eye, right_eye, nose, left_mouth, right_mouth],
            dtype=np.float32
        )

    def align_and_crop(self, image, landmarks_5):
        try:
            tform, _ = cv2.estimateAffinePartial2D(landmarks_5, self.ref_pts)

            if tform is None:
                return cv2.resize(image, self.arcface_input_size)

            return cv2.warpAffine(
                image,
                tform,
                self.arcface_input_size,
                borderMode=cv2.BORDER_REPLICATE
            )
        except Exception:
            return cv2.resize(image, self.arcface_input_size)

    def run_arcface(self, aligned_img):
        rgb_img = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)
        inp = (rgb_img.astype(np.float32) - 127.5) / 128.0
        inp = np.expand_dims(np.transpose(inp, (2, 0, 1)), axis=0)

        return self.arcface_sess.run(
            None,
            {self.arcface_input_name: inp}
        )[0]

    def extract_embedding_from_image(self, image: np.ndarray):
        """
        Exact API version of your batch enrollment embedding logic.
        Input must be BGR image from cv2.imdecode or cv2.imread.
        """

        t0 = time.perf_counter()

        # 1. YOLO detection
        results = self.detector(image, verbose=False)[0]

        face_crop = None

        for box in results.boxes:
            if int(box.cls) == self.face_class_id:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_crop = image[max(0, y1):y2, max(0, x1):x2]
                break

        # IMPORTANT: same fallback as batch code
        # Strict API validation: Reject if no face is found
        if face_crop is None or face_crop.size == 0:
            raise ValueError("No face detected in the uploaded image.")

        orig_h, orig_w = face_crop.shape[:2]

        t1 = time.perf_counter()

        # 2. Landmarks
        land_inp, tw, th = self.preprocess_landmarks(face_crop)

        self.landmark_interpreter.set_tensor(
            self.land_input_details[0]["index"],
            land_inp
        )
        self.landmark_interpreter.invoke()

        land_res = self.landmark_interpreter.get_tensor(
            self.land_output_details[0]["index"]
        ).flatten()

        num_coords = len(land_res)
        landmarks = None

        if num_coords == 136:
            landmarks = land_res.reshape(68, 2)

        elif num_coords == 1404:
            landmarks = land_res.reshape(468, 3)[:, :2]

        if landmarks is not None:
            scale_x = orig_w / float(tw)
            scale_y = orig_h / float(th)

            if np.max(landmarks) <= 1.0:
                landmarks[:, 0] *= tw * scale_x
                landmarks[:, 1] *= th * scale_y
            else:
                landmarks[:, 0] *= scale_x
                landmarks[:, 1] *= scale_y

            if num_coords == 136:
                pts5 = self.extract_5_points_from_68(landmarks)
            else:
                pts5 = self.extract_5_points_from_468(landmarks)

            aligned = self.align_and_crop(face_crop, pts5)

        else:
            aligned = cv2.resize(face_crop, self.arcface_input_size)

        t2 = time.perf_counter()

        # 3. ArcFace
        raw_embedding = self.run_arcface(aligned).flatten()

        norm = np.linalg.norm(raw_embedding)
        embedding = raw_embedding / norm if norm > 0 else raw_embedding

        t3 = time.perf_counter()

        timings = {
            "yolo_detection_ms": round((t1 - t0) * 1000, 2),
            "landmark_align_ms": round((t2 - t1) * 1000, 2),
            "arcface_embed_ms": round((t3 - t2) * 1000, 2),
            "total_extraction_ms": round((t3 - t0) * 1000, 2),
            "landmark_output_size": int(num_coords)
        }

        return embedding, timings
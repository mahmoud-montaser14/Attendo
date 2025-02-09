import base64
import csv
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from datetime import datetime
import os
import requests
from flask_cors import CORS  # Import CORS
from tqdm import tqdm
import random

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
# Load models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_recognizer.yml")
emotion_model = tf.keras.models.load_model("model_check_85_196.h5")
CSV_FILE = "students.csv"
MODEL_PATH = "face_recognizer2.yml"


# Load student names from CSV
students_list = []
with open("students.csv", newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    students_list = [row[0] for row in reader]

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


def base64_to_image(base64_data):
    encoded_data = base64_data.split(',')[1]  # Remove metadata header
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img



def detect_emotion(image):
    """Detect emotion from an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    emotions = {label: 0 for label in emotion_labels}
    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face_roi, (196, 196)) / 255.0
        face_resized = np.expand_dims(face_resized, axis=[0, -1])
        predictions = emotion_model.predict(face_resized)
        emotion_index = np.argmax(predictions)
        emotions[emotion_labels[emotion_index]] += 1
    return emotions


def recognize_students(image):
    """Recognize multiple students from an image."""
    recognized_students = {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face_roi, (300, 300))
        label, confidence = recognizer.predict(face_resized)
        student_name = students_list[label - 1] if 0 < label <= len(students_list) else "Unknown"
        if confidence < 50:
            recognized_students[student_name] = timestamp
    return recognized_students

def augment_image(image):
    """Apply various augmentations to the image."""
    aug_images = [image]
    height, width = image.shape
    # Flip image
    if random.random() > 0.5:
        aug_images.append(cv2.flip(image, 1))
    # Rotation
    for angle in range(-10, 10, 1):
        M = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)
        aug_images.append(cv2.warpAffine(image, M, (width, height)))
    # Shear transformation
    for shear_factor in range(-5, 5, 1):
        shear_matrix = np.float32([[1, shear_factor / 100, 0], [0, 1, 0]])
        aug_images.append(cv2.warpAffine(image, shear_matrix, (width, height)))
    # Adjust brightness
    for alpha in np.arange(0.8, 1.3, 0.1):
        adjusted = cv2.convertScaleAbs(image, alpha=alpha)
        aug_images.append(adjusted)
    # # Add Gaussian blur and random noise
    # aug_images.append(cv2.GaussianBlur(image, (5, 5), 0))
    # noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    # aug_images.append(cv2.add(image, noise))
    return aug_images

def detect_faces(img):
    """Detect faces using Haarcascade with preprocessing."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces

def align_face(img, rect):
    """Align and crop the face from the image."""
    (x, y, w, h) = rect
    center = (int(x + w // 2), int(y + h // 2))
    M = cv2.getRotationMatrix2D(center, 0, 1)
    aligned_face = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return aligned_face[y:y+h, x:x+w]

def read_students_from_csv(csv_file):
    """Load student data from CSV for training."""
    students = []
    with open(csv_file, newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            name = row[0].strip()
            image_paths = [path.strip() for path in row[1:] if path.strip()]
            if image_paths:
                students.append((name, image_paths))
    return students

def train_lbph():
    """Train the LBPH face recognizer using images from CSV."""
    students = read_students_from_csv(CSV_FILE)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    images, labels = [], []
    label_map = {}
    for idx, (name, image_paths) in tqdm(enumerate(students, start=1), total=len(students), desc="Training Model"):
        for image_path in image_paths:
            if not os.path.exists(image_path):
                print(f"[ERROR] Image not found for {name}: {image_path}")
                continue
            img = cv2.imread(image_path)
            if img is None:
                print(f"[ERROR] Could not read image for {name}.")
                continue
            faces = detect_faces(img)
            if len(faces) == 0:
                print(f"[WARNING] No face detected for {name} in {image_path}. Skipping...")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            for (x, y, w, h) in faces:
                face = align_face(gray, (x, y, w, h))
                face_resized = cv2.resize(face, (300, 300), interpolation=cv2.INTER_LINEAR)
                augmented_faces = augment_image(face_resized)
                for aug_face in augmented_faces:
                    images.append(np.array(aug_face, dtype=np.uint8))
                    labels.append(idx)
                    # Visualize augmentation briefly
                    # cv2.imshow("Augmented Face", aug_face)
                    # cv2.waitKey(100)
        label_map[idx] = name
    # cv2.destroyAllWindows()
    if images and labels:
        recognizer.train(images, np.array(labels, dtype=np.int32))
        recognizer.save(MODEL_PATH)
        print("[INFO] Training complete. Model saved.")
    else:
        print("[ERROR] No valid training data found. Training skipped.")

# @app.after_request
# def add_cors_headers(response):
#     """Ensure CORS headers are included in every response."""
#     response.headers["Access-Control-Allow-Origin"] = "http://localhost:4200"
#     response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
#     response.headers["Access-Control-Allow-Methods"] = "OPTIONS, GET, POST, PUT, DELETE"
#     return response

@app.route("/take_attendance", methods=["POST"])
def take_attendance():
    # """Endpoint to take attendance from an image."""
    # if request.method == "OPTIONS":
    #     response = jsonify({"message": "CORS preflight successful"})
    #     response.status_code = 200
    #     return response
    
    data = request.get_json()
    image_base64 = data.get("image")
    if not image_base64:
        return jsonify({"error": "No image provided"}), 400

    image = base64_to_image(image_base64)
    if image is None:
        return jsonify({"error": "Invalid image data"}), 400

    recognized_students = recognize_students(image)

    # Define backend endpoint
    backend_url = "http://reco.runasp.net/api/Attendance/record"

    try:
        # Send data to backend
        response = requests.post(backend_url, json=recognized_students)
        if response.status_code == 200:
            return jsonify(recognized_students), 200
        else:
            return jsonify({"no one":0}) #"error": "Failed to store attendance", "details": response.text}

    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Could not reach backend", "details": str(e)}), 500
    
    # return jsonify(recognized_students), 200


@app.route("/detect_emotion", methods=["POST"])
def detect_emotion_api():
    """Endpoint to detect emotion from an image."""
    data = request.get_json()
    image_base64 = data.get("image")
    if not image_base64:
        return jsonify({"error": "No image provided"}), 400

    image = base64_to_image(image_base64)
    if image is None:
        return jsonify({"error": "Invalid image data"}), 400

    emotions = detect_emotion(image)

    # Define backend endpoint
    backend_url = "http://reco.runasp.net/api/Emotion/detect"

    try:
        # Send data to backend
        response = requests.post(backend_url, json=emotions)
        if response.status_code == 200:
            return jsonify(emotions), 200
        else:
            return jsonify({"error": "Failed to store emotions", "details": response.text}), response.status_code

    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Could not reach backend", "details": str(e)}), 500
    
    # return jsonify(emotions), 200


@app.route("/train_model", methods=["POST"])
def train():
    train_lbph()
    return jsonify(f'Done, Model trained'), 200

# def train_model():
#     """Endpoint to train the LBPH face recognizer."""
#     try:
#         # Load student data from CSV
#         students = []
#         with open("students.csv", newline='', encoding='utf-8') as file:
#             reader = csv.reader(file)
#             next(reader)  # Skip header
#             for row in reader:
#                 name = row[0].strip()
#                 image_paths = [path.strip() for path in row[1:] if path.strip()]
#                 if image_paths:
#                     students.append((name, image_paths))

#         if not students:
#             return jsonify({"error": "No students found in CSV. Training aborted."}), 400

#         # Prepare training data
#         images = []
#         labels = []
#         label_map = {}

#         for idx, (name, image_paths) in enumerate(students, start=1):
#             for image_path in image_paths:
#                 if not os.path.exists(image_path):
#                     print(f"[ERROR] Image not found for {name}: {image_path}")
#                     continue

#                 img = cv2.imread(image_path)
#                 if img is None:
#                     print(f"[ERROR] Could not read image for {name}. Skipping...")
#                     continue

#                 # Detect faces
#                 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                 faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
#                 if len(faces) == 0:
#                     print(f"[WARNING] No face detected in {image_path}. Skipping...")
#                     continue

#                 # Align and resize face
#                 for (x, y, w, h) in faces:
#                     face_roi = gray[y:y + h, x:x + w]
#                     face_resized = cv2.resize(face_roi, (300, 300))
#                     images.append(face_resized)
#                     labels.append(idx)

#             label_map[idx] = name

#         if not images or not labels:
#             return jsonify({"error": "No valid training data found. Training aborted."}), 400

#         # Train the recognizer
#         recognizer.train(images, np.array(labels, dtype=np.int32))
#         recognizer.save("face_recognizerr.yml")

#         return jsonify({"message": "Model trained successfully"}), 200

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)

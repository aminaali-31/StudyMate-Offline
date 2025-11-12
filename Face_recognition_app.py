import face_recognition  as f
import cv2
import os
import sys
import numpy as np

class FaceRecognition:

    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('faces'):
            img_path = os.path.join('faces', image)
            face_image = f.load_image_file(img_path)
            enc = f.face_encodings(face_image)[0]
            name, _ = os.path.splitext(image)
            self.known_face_names.append(name)
            self.known_face_encodings.append(enc)

    def run_recognition(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            sys.exit("❌ Could not open camera")

        ret, frame = cap.read()
        cap.release()
        if not ret:
            return []

        # ✅ Resize first for speed
        frame_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = frame_small[:, :, ::-1]

        face_locations = f.face_locations(rgb_frame, model="hog")  # ✅ faster
        face_encodings = f.face_encodings(rgb_frame, face_locations)

        detected_names = []
        for face_encoding in face_encodings:
            distances = f.face_distance(self.known_face_encodings, face_encoding)
            best_index = np.argmin(distances)

            if distances[best_index] < 0.50:  # ✅ stricter threshold (faster stable matching)
                detected_names.append(self.known_face_names[best_index])
            else:
                detected_names.append("Unknown")

        return detected_names
    


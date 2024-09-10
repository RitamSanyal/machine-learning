import cv2
import face_recognition
import numpy as np


def load_and_encode_known_faces(known_faces_paths):
    known_face_encodings = []
    known_face_names = []

    for name, path in known_faces_paths.items():
        image = face_recognition.load_image_file(path)
        face_locations = face_recognition.face_locations(image)
        if face_locations:
            encodings = face_recognition.face_encodings(image, face_locations)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
            else:
                print(f"Could not compute face encoding for {name}.")
        else:
            print(f"No face detected in image for {name}.")

    return known_face_encodings, known_face_names

known_faces_paths = {
    "Ritam": "/home/ritam/PycharmProjects/machine-learning/Ritam.jpeg"
}

known_face_encodings, known_face_names = load_and_encode_known_faces(known_faces_paths)


def live_face_recognition(known_face_encodings, known_face_names):
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        print(f"Detected face locations: {face_locations}")

        if face_locations:
            try:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                print(f"Computed face encodings: {face_encodings}")
            except Exception as e:
                print(f"Error computing face encodings: {e}")
                continue
        else:
            print("No faces detected in the frame.")
            face_encodings = []

        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 30), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

        cv2.imshow('Webcam Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit
            break

    video_capture.release()
    cv2.destroyAllWindows()

live_face_recognition(known_face_encodings, known_face_names)
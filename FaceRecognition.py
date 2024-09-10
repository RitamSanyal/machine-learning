import cv2
import face_recognition
import numpy as np


# Load and encode known faces
def load_and_encode_known_faces(known_faces_paths):
    known_face_encodings = []
    known_face_names = []

    for name, path in known_faces_paths.items():
        # Load image using face_recognition and convert to RGB format
        image = face_recognition.load_image_file(path)
        # Detect face locations in the image
        face_locations = face_recognition.face_locations(image)
        # Ensure at least one face location is detected
        if face_locations:
            # Compute face encodings for the detected face
            encodings = face_recognition.face_encodings(image, face_locations)
            if encodings:  # Check if encoding is available
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
            else:
                print(f"Could not compute face encoding for {name}.")
        else:
            print(f"No face detected in image for {name}.")

    return known_face_encodings, known_face_names

# Provide paths to your known images
known_faces_paths = {
    "Ritam": "/home/ritam/PycharmProjects/machine-learning/Ritam.jpeg"
    # "Other_Person": "path_to_other_image.jpg"
}

# Load and encode known faces
known_face_encodings, known_face_names = load_and_encode_known_faces(known_faces_paths)


def live_face_recognition(known_face_encodings, known_face_names):
    # Initialize webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        print(f"Detected face locations: {face_locations}")

        # Ensure there are face locations to encode
        if face_locations:
            # Check if the frame data is in the correct format
            try:
                # Find all face encodings in the current frame
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                print(f"Computed face encodings: {face_encodings}")
            except Exception as e:
                print(f"Error computing face encodings: {e}")
                continue
        else:
            print("No faces detected in the frame.")
            face_encodings = []

        # Initialize array to hold names of recognized faces
        face_names = []

        for face_encoding in face_encodings:
            # See if the face matches any known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 30), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

        # Display the resulting frame
        cv2.imshow('Webcam Face Recognition', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    video_capture.release()
    cv2.destroyAllWindows()


# Run live face recognition
live_face_recognition(known_face_encodings, known_face_names)